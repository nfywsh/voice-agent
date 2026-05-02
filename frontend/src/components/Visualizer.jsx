// src/components/Visualizer.jsx
import React from 'react';
import { BarVisualizer } from '@livekit/components-react';

/**
 * 音频可视化组件
 * 直接使用 LiveKit 的 BarVisualizer 组件展示 Agent 的音频输出
 */
const Visualizer = () => {
  return (
    <div className="visualizer-wrapper">
      <BarVisualizer barCount={24} className="visualizer" />
      <p className="hint">AI 正在说话时会显示音频波形</p>
    </div>
  );
};

export default Visualizer;