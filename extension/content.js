chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "scrapeReviews") {
    const reviews = scrapeReviews();
    sendResponse({ reviews });
  }
  return true;
});

function scrapeReviews() {
  const reviews = [];

  const reviewBlocks = document.querySelectorAll('div[data-review-id]');

  reviewBlocks.forEach(block => {
    try {
      // try direct class first
      let textEl = block.querySelector('span[class*="wiI7pd"]');
      let text = '';

      if (textEl) {
        text = textEl.innerText.trim();
      } else {
        // fallback - grab all text from spans and join
        const allSpans = block.querySelectorAll('span');
        const parts = [];
        allSpans.forEach(span => {
          const t = span.innerText.trim();
          // skip dates, buttons, metadata
          if (
            t.length > 3 &&
            !t.includes('ago') &&
            !t.includes('Like') &&
            !t.includes('Share') &&
            !t.includes('More') &&
            !t.includes('Food:') &&
            !t.includes('Service:') &&
            !t.includes('Response') &&
            !span.children.length
          ) {
            parts.push(t);
          }
        });
        text = parts.join(' ').trim();
      }

      if (!text || text.length < 3) return;

      // get star rating
      const ratingEl = block.querySelector('span[class*="kvMYJc"]');
      let rating = 3;
      if (ratingEl) {
        const ariaLabel = ratingEl.getAttribute('aria-label') || '';
        const match = ariaLabel.match(/(\d)/);
        if (match) rating = parseInt(match[1]);
      }

      const dateEl = block.querySelector('span[class*="rsqaWe"]');
      const date = dateEl ? dateEl.innerText.trim() : '';

      reviews.push({ text, rating, date });

    } catch (e) {
      console.error('error:', e);
    }
  });

  return reviews;
}