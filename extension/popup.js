const analyzeBtn = document.getElementById('analyze-btn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const errorDiv = document.getElementById('error');

analyzeBtn.addEventListener('click', async () => {
  results.style.display = 'none';
  errorDiv.style.display = 'none';
  loading.style.display = 'block';
  analyzeBtn.disabled = true;

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    // inject content script manually
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ['content.js']
    });

    // wait for script to load
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

    const summaryResponse = await fetch('http://127.0.0.1:5000/summary', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ real_reviews: data.real_reviews })
    });

    const summaryData = await summaryResponse.json();

    displayResults(data, summaryData.summary);

  } catch (err) {
    console.error(err);
    loading.style.display = 'none';
    errorDiv.style.display = 'block';
    analyzeBtn.disabled = false;
  }
});

function displayResults(data, summary) {
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

  document.getElementById('summary-text').textContent = summary || 'Not enough real reviews to summarize.';

  const list = document.getElementById('review-list');
  list.innerHTML = '';

  data.reviews.forEach(r => {
    const labelColor =
      r.label === 'Real' ? 'badge-real' :
      r.label === 'Fake' ? 'badge-fake' : 'badge-suspicious';

    list.innerHTML += `
      <div class="review-item">
        <div class="review-label">
          <span class="badge ${labelColor}">${r.label}</span>
        </div>
        <div class="review-text">${r.text}</div>
      </div>
    `;
  });

  results.style.display = 'block';
}