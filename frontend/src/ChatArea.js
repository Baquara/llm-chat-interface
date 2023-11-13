// ChatArea.js

import React, { useState } from 'react';
import './ChatArea.css';
import ChatComponent from './ChatComponent';

function ChatArea() {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState("");

  const handleInputChange = (e) => {
    setUserInput(e.target.value);
  };

const handleSendClick = async () => {
  if (userInput.trim() === "") return;

  // Add the user's message to the conversation
  setMessages(prev => [...prev, { sender: "user", message: userInput }]);

  // Clear the input box
  setUserInput("");

  // Add an empty bot response placeholder with "..."
  setMessages(prev => [...prev, { sender: "bot", message: "..." }]);

  // Create data object for server
  const data = {
    user_message: userInput,
    username: "JohnDoe",
    botname: "ChatGPT",
    print_timings: true,
    in_code_block: false
  };

  // Send data to the server
  const response = await fetch("http://127.0.0.1:8000/generate_response/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });

  if (response.ok) {
    const reader = response.body.getReader();
    let chunks = "";
    reader.read().then(function processText({ done, value }) {
      const decodedChunk = new TextDecoder("utf-8").decode(value);
      chunks += decodedChunk;

      // Update bot's message with the new chunk
      setMessages(prev => {
        const messagesCopy = [...prev];
        if (messagesCopy.length > 0 && messagesCopy[messagesCopy.length - 1].sender === "bot") {
          messagesCopy[messagesCopy.length - 1].message = chunks.startsWith("...") ? chunks.substring(3) : chunks;
        } else {
          messagesCopy.push({ sender: "bot", message: decodedChunk });
        }
        return messagesCopy;
      });

      if (!done) {
        return reader.read().then(processText);
      }
    });
  } else {
    console.error("Failed to fetch from server");
  }
};

  return (
    <div className="chat-area">
      <div className="model-selector" id="customModelSelector">
        <select>
          <option>Select a model</option>
          {/* Model options would go here */}
        </select>
      </div>
      <ChatComponent messages={messages} />
      <div className="chat-inputs">
        <label htmlFor="fileInput" className="file-upload-button">
          <i className="fas fa-paperclip"></i>
        </label>
        <textarea
          placeholder="Send a message"
          value={userInput}
          onChange={handleInputChange}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault(); // This prevents the newline from being added
              handleSendClick();
            }
          }}
        ></textarea>
        <button onClick={handleSendClick} className="send-button">
          <i className="fas fa-paper-plane"></i>
        </button>
        <input type="file" id="fileInput" style={{ display: 'none' }} />
      </div>
    </div>
  );
}

export default ChatArea;
