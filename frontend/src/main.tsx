import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './style.css';
import './vite-env.d.ts';

createRoot(document.getElementById('root') as HTMLElement).render(<App />);
