// src/App.jsx
import React, { useState } from 'react';
import VoiceRoom from './components/VoiceRoom';
import '@livekit/components-styles';

function App() {
  const [roomName, setRoomName] = useState(`room-${Math.random().toString(36).slice(2, 8)}`);
  const [userName, setUserName] = useState(`User-${Math.random().toString(36).slice(2, 6)}`);
  const [joined, setJoined] = useState(false);

  const handleJoin = (e) => {
    e.preventDefault();
    if (roomName.trim() && userName.trim()) {
      setJoined(true);
    }
  };

  if (joined) {
    return (
      <VoiceRoom
        roomName={roomName}
        userName={userName}
        onLeave={() => setJoined(false)}
      />
    );
  }

  return (
    <div className="join-container">
      <div className="join-card">
        <h1>🎤 语音聊天机器人</h1>
        <p className="subtitle">全双工对话 · 支持打断 · 会唱歌</p>

        <form onSubmit={handleJoin}>
          <div className="form-group">
            <label htmlFor="roomName">房间名</label>
            <input
              id="roomName"
              type="text"
              value={roomName}
              onChange={(e) => setRoomName(e.target.value)}
              placeholder="输入房间名"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="userName">你的名字</label>
            <input
              id="userName"
              type="text"
              value={userName}
              onChange={(e) => setUserName(e.target.value)}
              placeholder="输入你的名字"
              required
            />
          </div>

          <button type="submit" className="join-btn">
            加入房间
          </button>
        </form>

        <div className="features">
          <span>🎵 会唱歌</span>
          <span>⚡ 低延迟</span>
          <span>🗣️ 全双工</span>
        </div>
      </div>
    </div>
  );
}

export default App;