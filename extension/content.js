// listen for message from popup
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
      const textEl = block.querySelector('span[class*="wiI7pd"]');
      const ratingEl = block.querySelector('span[class*="kvMYJc"]');

      if (!textEl) return;

      const text = textEl.innerText.trim();
      if (!text || text.length === 0) return;

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
      console.error('error scraping review:', e);
    }
  });

  return reviews;
}