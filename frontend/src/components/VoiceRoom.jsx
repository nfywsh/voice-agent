// src/components/VoiceRoom.jsx
import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  LiveKitRoom,
  RoomAudioRenderer,
  ControlBar,
  BarVisualizer,
  useVoiceAssistant,
  useRoomContext,
  useLocalParticipant,
  useDataChannel,
} from '@livekit/components-react';
import { Track, ConnectionState, DataPacket_Kind } from 'livekit-client';

// ============ Token 获取 ============

async function getToken(roomName, userName) {
  const response = await fetch(`/api/token?room=${encodeURIComponent(roomName)}&username=${encodeURIComponent(userName)}`);
  if (!response.ok) {
    throw new Error(`Failed to get token: ${response.status}`);
  }
  const data = await response.json();
  return data.token;
}

// ============ 对话转写组件 ============

function TranscriptionPanel() {
  const [messages, setMessages] = useState([]);
  const messagesEndRef = useRef(null);

  // 监听 voice assistant 状态变化
  const { state, agent } = useVoiceAssistant();

  // 监听 data channel 消息（Agent 推送的文本消息，用于 TTS 降级时展示）
  useDataChannel((message) => {
    try {
      const data = JSON.parse(new TextDecoder().decode(message.payload));
      if (data.type === 'transcript') {
        setMessages((prev) => [
          ...prev.slice(-50), // 保留最近 50 条
          {
            id: Date.now(),
            role: data.role || 'agent',
            text: data.text,
            time: new Date().toLocaleTimeString(),
          },
        ]);
      }
    } catch {
      // 非 JSON 消息忽略
    }
  });

  // Agent 说话时显示转写
  useEffect(() => {
    if (state === 'speaking' && agent) {
      // voice assistant SDK 会自动管理转写
    }
  }, [state, agent]);

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="transcription-panel">
      <h3>对话记录</h3>
      <div className="messages">
        {messages.length === 0 && (
          <p className="hint">对话内容将显示在这里...</p>
        )}
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.role}`}>
            <span className="role">{msg.role === 'agent' ? '🤖 AI' : '👤 你'}</span>
            <span className="text">{msg.text}</span>
            <span className="time">{msg.time}</span>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}

// ============ AI 状态指示器 ============

function AgentStatus() {
  const { state } = useVoiceAssistant();

  const statusConfig = {
    listening: { label: '正在聆听...', color: '#4CAF50', icon: '👂' },
    thinking: { label: '思考中...', color: '#FF9800', icon: '🤔' },
    speaking: { label: '正在说话...', color: '#2196F3', icon: '🗣️' },
    connecting: { label: '连接中...', color: '#9E9E9E', icon: '🔄' },
    disconnected: { label: '未连接', color: '#F44336', icon: '❌' },
  };

  // 默认状态
  const config = statusConfig[state] || statusConfig.disconnected;

  return (
    <div className="agent-status">
      <span className="status-dot" style={{ backgroundColor: config.color }}></span>
      <span className="status-icon">{config.icon}</span>
      <span className="status-label">{config.label}</span>
    </div>
  );
}

// ============ 错误处理组件 ============

function ErrorDisplay({ error, onRetry }) {
  return (
    <div className="error-container">
      <h2>⚠️ 连接出现问题</h2>
      <p className="error-message">{error}</p>
      <button onClick={onRetry} className="retry-btn">重新连接</button>
    </div>
  );
}

// ============ 主语音房间组件 ============

function VoiceRoomInner({ roomName, userName, onLeave }) {
  const room = useRoomContext();
  const [connectionState, setConnectionState] = useState(ConnectionState.Connecting);
  const [error, setError] = useState(null);

  // 监听连接状态
  useEffect(() => {
    const handleStateChange = (newState) => {
      setConnectionState(newState);
      if (newState === ConnectionState.Disconnected) {
        // 可以自动重连或提示用户
      }
    };

    room.on('connection_state_changed', handleStateChange);
    return () => {
      room.off('connection_state_changed', handleStateChange);
    };
  }, [room]);

  return (
    <div className="room-container">
      <div className="room-header">
        <h2>🎤 语音助手 · 房间: {roomName}</h2>
        <AgentStatus />
      </div>

      <div className="room-content">
        <div className="visualizer-section">
          <BarVisualizer barCount={20} className="visualizer" />
          <p className="hint">点击下方按钮开始对话，你可以随时打断 AI</p>
        </div>

        <TranscriptionPanel />
      </div>

      <div className="room-footer">
        <ControlBar variation="verbose" className="control-bar" />
        <button onClick={onLeave} className="leave-btn">离开房间</button>
      </div>

      <RoomAudioRenderer />
    </div>
  );
}

// ============ 语音房间入口 ============

function VoiceRoom({ roomName, userName, onLeave }) {
  const [token, setToken] = useState(null);
  const [connecting, setConnecting] = useState(true);
  const [error, setError] = useState(null);

  const livekitUrl = import.meta.env.VITE_LIVEKIT_URL || 'wss://localhost:7880';

  // 获取 Token
  useEffect(() => {
    let mounted = true;

    getToken(roomName, userName)
      .then((t) => {
        if (mounted) {
          setToken(t);
          setConnecting(false);
        }
      })
      .catch((err) => {
        console.error('Token error:', err);
        if (mounted) {
          setError('无法连接到服务器，请检查网络或刷新页面重试');
          setConnecting(false);
        }
      });

    return () => { mounted = false; };
  }, [roomName, userName]);

  // 重试
  const handleRetry = useCallback(() => {
    setError(null);
    setConnecting(true);
    getToken(roomName, userName)
      .then((t) => {
        setToken(t);
        setConnecting(false);
      })
      .catch((err) => {
        console.error('Token retry error:', err);
        setError('重试失败，请检查服务器状态');
        setConnecting(false);
      });
  }, [roomName, userName]);

  // 错误状态
  if (error) {
    return <ErrorDisplay error={error} onRetry={handleRetry} />;
  }

  // 加载中
  if (connecting || !token) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>正在连接语音服务...</p>
      </div>
    );
  }

  // 连接房间
  return (
    <LiveKitRoom
      serverUrl={livekitUrl}
      token={token}
      connect={true}
      audio={true}
      video={false}
      onDisconnected={() => {
        console.log('Disconnected from room');
        onLeave();
      }}
      onError={(err) => {
        console.error('Room error:', err);
        setError(`房间连接错误: ${err.message}`);
      }}
      dataPacketType={DataPacket_Kind.RELIABLE}
    >
      <VoiceRoomInner
        roomName={roomName}
        userName={userName}
        onLeave={onLeave}
      />
    </LiveKitRoom>
  );
}

export default VoiceRoom;