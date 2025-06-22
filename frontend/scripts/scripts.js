// TODO : refactor this file to separate multiple functionalities and base content in multiple files.

// Store the associated tab conversations in a JS object.
const tabConversations = {
    welcome: `
        <div class="tab-conversation">
            <div class="message user">Hello there !</div>
            <div class="message assistant">Hi!! You're back! 🎉 Welcome to your message archive — home of every curious musing, spiral, victory dance, and late-night tangent you’ve ever shared.<br>Ready to do some exploring?</div>
            <div class="message user">Sure ! What can I do here again ?</div>
            <div class="message assistant">Glad you asked! This space lets you sift through past chats by meaning, tone, thread... you name it.<br>The messages on the right? They’re yours.<br>The ones on the left? That’s me, helping you navigate 💡 And mostly, serving as a journal, note-taker, and little encouragement machine ;)</div>
            <div class="message user">That's kinda awesome. <strong>Surely a responsible dev would've stored more useful information in the "Help" tab.</strong></div>
        </div>
    `,
    help: `
        <div class="tab-conversation">
            <div class="message user">Okay assistant, I’m overwhelmed. What now?</div>
            <div class="message assistant">Take a breath — I got you 🫶<br>Here’s your toolkit:<br><br>• Semantic search — type in anything, from “when was I hopeful” to “why do I always forget the rice.”<br>• Filter by convo — stick to one thread.<br>• Filter by tone — joy, angst, silliness, wisdom... all tagged.<br>• Etc... — because of course there’s more coming soon™ 😌</div>
            <div class="message user">And if I click on a message?</div>
            <div class="message assistant"> Boom 💥 Full message details. You can update metadata, re-tag, or just check the date. It’s your space to explore, remember, and organize gently 🧷</div>
            <div class="message user">Sure. WHEN I'll implement it ^^'</div>
        </div>
    `,
    credits: `
        <div class="tab-conversation">
            <div class="message user"> Hey, who made this? It’s kinda... sleek.</div>
            <div class="message assistant"> Why, you did! With your own hands 🛠️<br>From scratch: HTML, CSS, JavaScript, semantic search — all bundled into this little memory garden.<br>(And some help from Copilot, me, and a little Linux magic.)</div>
            <div class="message user">Oh... I guess I <i>did</i>. That’s kinda cool.<br>And you, what are you doing in there?</div>
            <div class="message assistant">  Living the digital dream 💻<br>Well — not living, more like being stored, and made conveniently searchable.<br>Which is already pretty rad.</div>
        </div>
    `,
    lorem: `
        <div class="tab-conversation">
            <div class="message assistant">Lorem ipsum dolor sit amet, consectetur adipiscing elit.</div>
            <div class="message user">Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.</div>
            <div class="message assistant">Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.</div>
            <div class="message user">Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.</div>
            <div class="message assistant">Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</div>
            <div class="message user">
                Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque euismod, nisi eu consectetur consectetur, nisl nisi consectetur nisi, euismod euismod nisi nisi euismod. 
                Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Etiam euismod, nisi eu consectetur consectetur, nisl nisi consectetur nisi, euismod euismod nisi nisi euismod. 
                Nullam ac urna eu felis dapibus condimentum sit amet a augue. Sed non neque elit. Sed ut imperdiet nisi. Proin condimentum fermentum nunc. Etiam pharetra, erat sed fermentum feugiat, velit mauris egestas quam, ut aliquam massa nisl quis neque. 
                Suspendisse in orci enim. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. 
                Mauris placerat eleifend leo. Quisque sit amet est et sapien ullamcorper pharetra. Vestibulum erat wisi, condimentum sed, commodo vitae, ornare sit amet, wisi. 
                Aenean fermentum, elit eget tincidunt condimentum, eros ipsum rutrum orci, sagittis tempus lacus enim ac dui. 
                Donec non enim in turpis pulvinar facilisis. Ut felis. Praesent dapibus, neque id cursus faucibus, tortor neque egestas augue, eu vulputate magna eros eu erat. 
                Aliquam erat volutpat. Nam dui mi, tincidunt quis, accumsan porttitor, facilisis luctus, metus. 
                Phasellus ultrices nulla quis nibh. Quisque a lectus. Donec consectetuer ligula vulputate sem tristique cursus. 
                Nam nulla quam, gravida non, commodo a, sodales sit amet, nisi. Pellentesque fermentum dolor. 
                Aliquam quam lectus, facilisis auctor, ultrices ut, elementum vulputate, nunc. 
                Sed adipiscing ornare risus. Morbi est est, blandit sit amet, sagittis vel, euismod vel, velit. 
                Pellentesque egestas sem. Suspendisse commodo ullamcorper magna. 
                Nullam porttitor lacus at turpis. Donec posuere augue in quam. Etiam vel tortor sodales tellus ultricies commodo. 
                Suspendisse potenti. Aenean in sem ac leo mollis blandit. 
                Donec neque quam, dignissim in, mollis nec, sagittis eu, wisi. 
                Morbi nec metus. Donec et nunc. Nam eget dui. 
                Etiam rhoncus. Maecenas tempus, tellus eget condimentum rhoncus, sem quam semper libero, sit amet adipiscing sem neque sed ipsum. 
                Nam quam nunc, blandit vel, luctus pulvinar, hendrerit id, lorem. 
                Maecenas nec odio et ante tincidunt tempus. Donec vitae sapien ut libero venenatis faucibus. 
                Nullam quis ante. Etiam sit amet orci eget eros faucibus tincidunt. Duis leo. Sed fringilla mauris sit amet nibh. 
                Donec sodales sagittis magna. Sed consequat, leo eget bibendum sodales, augue velit cursus nunc, quis gravida magna mi a libero. 
                Fusce vulputate eleifend sapien. Vestibulum purus quam, scelerisque ut, mollis sed, nonummy id, metus. 
                Nullam accumsan lorem in dui. Cras ultricies mi eu turpis hendrerit fringilla. 
                Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; In ac dui quis mi consectetuer lacinia. 
                Nam pretium turpis et arcu. Duis arcu tortor, suscipit eget, imperdiet nec, imperdiet iaculis, ipsum. 
                Sed aliquam ultrices mauris. Integer ante arcu, accumsan a, consectetuer eget, posuere ut, mauris. 
                Praesent adipiscing. Phasellus ullamcorper ipsum rutrum nunc. 
                Nunc nonummy metus. Vestibulum volutpat pretium libero. 
                Cras id dui. Aenean ut eros et nisl sagittis vestibulum. 
                Nullam nulla eros, ultricies sit amet, nonummy id, imperdiet feugiat, pede. 
                Sed lectus. Donec mollis hendrerit risus. Phasellus nec sem in justo pellentesque facilisis. 
                Etiam imperdiet imperdiet orci. Nunc nec neque. 
                Phasellus leo dolor, tempus non, auctor et, hendrerit quis, nisi. 
                Curabitur ligula sapien, tincidunt non, euismod vitae, posuere imperdiet, leo. 
                Maecenas malesuada. Praesent congue erat at massa. 
                Sed cursus turpis vitae tortor. Donec posuere vulputate arcu. 
                Phasellus accumsan cursus velit. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Sed aliquam, nisi quis porttitor congue, elit erat euismod orci, ac placerat dolor lectus quis orci. 
                Phasellus consectetuer vestibulum elit. Aenean tellus metus, bibendum sed, posuere ac, mattis non, nunc. 
                Vestibulum fringilla pede sit amet augue. In turpis. Pellentesque posuere. 
                Praesent turpis. Aenean posuere, tortor sed cursus feugiat, nunc augue blandit nunc, eu sollicitudin urna dolor sagittis lacus. 
                Donec elit libero, sodales nec, volutpat a, suscipit non, turpis. Nullam sagittis. 
                Suspendisse pulvinar, augue ac venenatis condimentum, sem libero volutpat nibh, nec pellentesque velit pede quis nunc. 
                Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Fusce id purus. 
                Aliquam erat volutpat. Pellentesque sagittis, sem sit amet interdum ultrices, lacus sem dictum enim, quis dictum massa enim nec sem. 
                Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. 
                In dui magna, posuere eget, vestibulum et, tempor auctor, justo. In ac felis quis tortor malesuada pretium. 
                Pellentesque auctor neque nec urna. Proin sapien ipsum, porta a, auctor quis, euismod ut, mi. 
                Aenean viverra rhoncus pede. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. 
                Ut non enim eleifend felis pretium feugiat. Vivamus quis mi. 
                Phasellus a est. Phasellus magna. In hac habitasse platea dictumst. 
                Curabitur at lacus ac velit ornare lobortis. Curabitur a felis in nunc fringilla tristique. 
                Morbi mattis ullamcorper velit. Phasellus gravida semper nisi. 
                Nullam vel sem. Pellentesque libero tortor, tincidunt et, tincidunt eget, semper nec, quam. 
                Sed hendrerit. Morbi ac felis. Nunc egestas, augue at pellentesque laoreet, felis eros vehicula leo, at malesuada velit leo quis pede. 
                Donec interdum, metus et hendrerit aliquet, dolor diam sagittis ligula, eget egestas libero turpis vel mi. 
                Fusce ac felis sit amet ligula pharetra condimentum. 
                Mauris sollicitudin fermentum libero. Praesent nonummy mi in odio. 
                Nunc interdum lacus sit amet orci. Vestibulum rutrum, mi nec elementum vehicula, eros quam gravida nisl, id fringilla neque ante vel mi. 
                Morbi mollis tellus ac sapien. Phasellus volutpat, metus eget egestas mollis, lacus lacus blandit dui, id egestas quam mauris ut lacus. 
                Fusce vel dui. Pellentesque egestas, neque sit amet convallis pulvinar, justo nulla eleifend augue, ac auctor orci leo non est. 
                Ut leo. Suspendisse potenti. 
            </div>
        </div>
    `,
};

function switchTab(e) {
    const clickedButton = e.currentTarget;
    // Reset all tab-link buttons to the --highlight color.
    document.querySelectorAll('.tab-link').forEach(button => {
        button.style.backgroundColor = getComputedStyle(document.documentElement)
            .getPropertyValue('--highlight');
        button.style.borderTop = '';
    });
    // Set the clicked button to the --bg color.
    clickedButton.style.backgroundColor = getComputedStyle(document.documentElement)
        .getPropertyValue('--bg');
    clickedButton.style.borderTop = '2px solid var(--accent)';
    // Get the tab key from a data attribute.
    const tabKey = clickedButton.getAttribute('data-tab');

    // Retrieve the conversation HTML. (Fallback if no conversation is defined.)
    const conversationHTML = tabConversations[tabKey] || `
        <div class="tab-conversation">
            <div class="message assistant">No conversation available.</div>
        </div>
    `;
    // Replace the content in the active-viewer with the selected conversation.
    document.getElementById('active-viewer').innerHTML = conversationHTML;
}

// Optionally, add event listeners to all tab-link buttons on page load.
document.querySelectorAll('.tab-link').forEach(button => {
    button.addEventListener('click', switchTab);
});

// Trigger default tab (if required) after the listeners are set.
document.getElementById("defaultHelp").click();

