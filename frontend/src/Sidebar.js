import React, { useState, useRef } from 'react';
import './Sidebar.css';

function Sidebar() {
  const [visible, setVisible] = useState(true);
  const isResizing = useRef(false); // Using useRef instead of useState
  const [startX, setStartX] = useState(0);

  const handleMouseDown = (event) => {
    console.log("Mousedown triggered");
    isResizing.current = true; // Setting the ref directly
    console.log('isResizing after set:', isResizing.current);
    setStartX(event.clientX);
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  const handleMouseMove = (event) => {
    console.log('Mousemove triggered');
    console.log('isResizing state in handleMouseMove:', isResizing.current);

    if (!isResizing.current) {
      console.log('Resizing not enabled, returning.');
      return;
    }

    const currentX = event.clientX;

    if (currentX > startX) {
      console.log('Dragging right');
    } else if (currentX < startX) {
      console.log('Dragging left');
    }

    document.querySelector('.sidebar').style.width = currentX + "px";
  };

  const handleMouseUp = () => {
    console.log("Mouseup triggered");
    isResizing.current = false;
    console.log('isResizing after reset:', isResizing.current);
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  };
 // Mock data for chat sessions
  const mockChats = [
    "Chat Component Positioning Issue",
    "Display <img> tags correctly",
    "Get First Prompt Content",
    "Función para generar respuestas",
    "Maltese Puppy Hyperrealism Art",
    "New chat",
    "Streaming FastAPI Responses",
    "FastAPI Chatbot Endpoint",
    "Chat Component Simulation",
    "Image Elements Described"
  ];

  return (
    <div className="sidebar" style={{ width: visible ? 'auto' : '40px' }}>
      <button className="toggle-btn" onClick={() => setVisible(!visible)}>{visible ? '👁️' : '👁️'}</button>
      <div className="sidebar-content" style={{ display: visible ? 'block' : 'none' }}>
        <button>+ New Chat</button>
        <input type="text" placeholder="Search conversation..." />

        {/* Mapping over mock data to render chat items */}
        <ul className="chat-list">
          {mockChats.map((chat, index) => (
            <li key={index} className="chat-item">
              <svg className="chat-icon" viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
              <div className="chat-text">{chat}</div>
            </li>
          ))}
        </ul>

        <div className="resizer" onMouseDown={handleMouseDown}></div>
      </div>
    </div>
  );
}

export default Sidebar;
