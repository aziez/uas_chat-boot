/**
 * Toko Sembako Barokah - Chatbot Client Scripts
 * -----------------------------------------------
 * Auto-focus chat input after every Streamlit rerun.
 * Loaded via: components.html(f"<script>{Path('scripts.js').read_text()}</script>", height=0)
 */

(function () {
    'use strict';

    var parentDoc = window.parent.document;

    /**
     * Locate the chat input textarea and focus it.
     * Tries the official data-testid first, then falls back to class selectors.
     */
    function focusChatInput() {
        var selectors = [
            'textarea[data-testid="stChatInputTextArea"]',
            '.stChatInput textarea',
            '[data-testid="stChatInput"] textarea'
        ];

        var input = null;
        for (var i = 0; i < selectors.length; i++) {
            var matches = parentDoc.querySelectorAll(selectors[i]);
            if (matches.length > 0) {
                input = matches[matches.length - 1]; // last on page
                break;
            }
        }

        if (input && document.activeElement !== input) {
            input.focus();
        }
    }

    // Initial focus after Streamlit finishes rendering
    setTimeout(focusChatInput, 150);

    // Re-focus if Streamlit recreates the input element
    var observer = new MutationObserver(function () {
        setTimeout(focusChatInput, 100);
    });

    observer.observe(parentDoc.body, { childList: true, subtree: true });

    // Disconnect after 5s to avoid long-running observer overhead
    setTimeout(function () {
        observer.disconnect();
    }, 5000);
})();
