import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import History from './components/History';
import ModelComparison from './components/ModelComparison';
import './App.css';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <Navbar />
        
        <main className="max-w-7xl mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/history" element={<History />} />
            <Route path="/models" element={<ModelComparison />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="bg-white border-t border-gray-200 mt-12">
          <div className="max-w-7xl mx-auto px-4 py-6 text-center text-gray-600 text-sm">
            <p>Premier League Match Predictor - FYP 2025-2026</p>
            <p className="mt-1">ATU Galway-Mayo | BSc (Honours) Computing</p>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;