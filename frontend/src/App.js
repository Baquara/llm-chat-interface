import React from 'react';
import './App.css';
import Sidebar from './Sidebar';
import ChatArea from './ChatArea';

function App() {
  return (
    <div className="app">
      <Sidebar />
      <ChatArea />
    </div>
  );
}

export default App;
