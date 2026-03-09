if (!window.__reviewAnalyzerInjected) {
  window.__reviewAnalyzerInjected = true;

  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "scrapeReviews") {
      const reviews = scrapeReviews();
      sendResponse({ reviews });
    }
    return true;
  });
}

function scrapeReviews() {
  const reviews = [];
  const seen = new Set();

  const reviewBlocks = document.querySelectorAll('div[data-review-id]');

  reviewBlocks.forEach(block => {
    try {
      let textEl = block.querySelector('span[class*="wiI7pd"]');
      let text = '';

      if (textEl) {
        text = textEl.innerText.trim();
      } else {
        const allSpans = block.querySelectorAll('span');
        const parts = [];
        allSpans.forEach(span => {
          const t = span.innerText.trim();
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

      // skip duplicates
      if (seen.has(text)) return;
      seen.add(text);

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