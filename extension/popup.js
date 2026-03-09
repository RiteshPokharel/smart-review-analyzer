const analyzeBtn = document.getElementById('analyze-btn');
const summaryBtn = document.getElementById('summary-btn');
const loading = document.getElementById('loading');
const summaryLoading = document.getElementById('summary-loading');
const results = document.getElementById('results');
const errorDiv = document.getElementById('error');

let storedRealReviews = [];

analyzeBtn.addEventListener('click', async () => {
  results.style.display = 'none';
  errorDiv.style.display = 'none';
  loading.style.display = 'block';
  analyzeBtn.disabled = true;
  document.getElementById('summary-text').textContent = '';
  summaryBtn.style.display = 'block';
  summaryBtn.disabled = false;
  summaryBtn.textContent = 'Generate Summary';

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ['content.js']
    });

    await new Promise(resolve => setTimeout(resolve, 500));

    const response = await chrome.tabs.sendMessage(tab.id, { action: 'scrapeReviews' });
    const reviews = response?.reviews || [];

    if (reviews.length === 0) {
      throw new Error('no reviews found');
    }

    const apiResponse = await fetch('http://127.0.0.1:5000/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviews })
    });

    const data = await apiResponse.json();

    storedRealReviews = data.real_reviews;

    displayResults(data);

  } catch (err) {
    console.error(err);
    loading.style.display = 'none';
    errorDiv.style.display = 'block';
    analyzeBtn.disabled = false;
  }
});

summaryBtn.addEventListener('click', async () => {
  summaryBtn.disabled = true;
  summaryBtn.textContent = 'Generating...';
  summaryLoading.style.display = 'block';
  document.getElementById('summary-text').textContent = '';

  try {
    const summaryResponse = await fetch('http://127.0.0.1:5000/summary', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ real_reviews: storedRealReviews })
    });

    const summaryData = await summaryResponse.json();

    summaryLoading.style.display = 'none';
    document.getElementById('summary-text').textContent = summaryData.summary || 'Not enough real reviews to summarize.';
    summaryBtn.style.display = 'none';

  } catch (err) {
    console.error(err);
    summaryLoading.style.display = 'none';
    summaryBtn.disabled = false;
    summaryBtn.textContent = 'Generate Summary';
  }
});

function displayResults(data) {
  loading.style.display = 'none';
  analyzeBtn.disabled = false;

  document.getElementById('real-count').textContent = data.counts.real;
  document.getElementById('fake-count').textContent = data.counts.fake;
  document.getElementById('suspicious-count').textContent = data.counts.suspicious;

  const pos = data.sentiment.positive;
  const neg = data.sentiment.negative;
  document.getElementById('positive-pct').textContent = pos + '%';
  document.getElementById('negative-pct').textContent = neg + '%';
  document.getElementById('positive-bar').style.width = pos + '%';
  document.getElementById('negative-bar').style.width = neg + '%';

  const list = document.getElementById('review-list');
  list.innerHTML = '';

  data.reviews.forEach((r, index) => {
    const labelClass =
      r.label === 'Real' ? 'pill-real' :
      r.label === 'Fake' ? 'pill-fake' : 'pill-suspicious';

    const sentimentClass =
      r.sentiment === 'Positive' ? 'pill-positive' :
      r.sentiment === 'Negative' ? 'pill-negative' : 'pill-neutral';

    const isLong = r.text.length > 120;
    const textId = `review-text-${index}`;
    const btnId = `read-more-${index}`;

    list.innerHTML += `
      <div class="review-item">
        <div class="review-meta">
          <span class="pill ${labelClass}">${r.label}</span>
          <span class="pill ${sentimentClass}">${r.sentiment}</span>
        </div>
        <div class="review-text collapsed" id="${textId}">${r.text}</div>
        ${isLong ? `<span class="read-more" id="${btnId}">Read more</span>` : ''}
      </div>
    `;
  });

  data.reviews.forEach((r, index) => {
    if (r.text.length > 120) {
      const btn = document.getElementById(`read-more-${index}`);
      const textEl = document.getElementById(`review-text-${index}`);
      if (btn) {
        btn.addEventListener('click', () => {
          if (textEl.classList.contains('collapsed')) {
            textEl.classList.remove('collapsed');
            btn.textContent = 'Show less';
          } else {
            textEl.classList.add('collapsed');
            btn.textContent = 'Read more';
          }
        });
      }
    }
  });

  results.style.display = 'block';
}