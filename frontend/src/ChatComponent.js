import React from 'react';
import './ChatComponent.css';

function ChatComponent({ messages }) {
  return (
    <div className="chat-component">
      {messages.map((msg, index) => (
        <div key={index} className={`message ${msg.sender}`}>
          <p dangerouslySetInnerHTML={{ __html: msg.message }}></p>
        </div>
      ))}
    </div>
  );
}

export default ChatComponent;
