/**
 * Toko Sembako Barokah - Chatbot Client Scripts
 * Auto-focus chat input after every Streamlit rerun.
 * Note: st.html() is NOT iframed, so we use document directly.
 */

(function () {
    'use strict';

    function focusChatInput() {
        var selectors = [
            'textarea[data-testid="stChatInputTextArea"]',
            '.stChatInput textarea',
            '[data-testid="stChatInput"] textarea'
        ];

        var input = null;
        for (var i = 0; i < selectors.length; i++) {
            var matches = document.querySelectorAll(selectors[i]);
            if (matches.length > 0) {
                input = matches[matches.length - 1];
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

    observer.observe(document.body, { childList: true, subtree: true });

    // Disconnect after 5s to avoid long-running observer overhead
    setTimeout(function () {
        observer.disconnect();
    }, 5000);
})();
