import React, { useState, useEffect } from 'react';
import GameweekPredictions from './GameweekPredictions';
import { api } from '../services/api';

const Dashboard = () => {
  const [stats, setStats] = useState({
    accuracy: 0,
    correct: 0,
    total: 0
  });
  const [models, setModels] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      setLoading(true);
      const [modelsData] = await Promise.all([
        api.getModels()
      ]);
      
      setStats({ accuracy: 50, correct: 10, total: 20 }); // Temp values
      setModels(modelsData);
    } catch (err) {
      console.error('Failed to load stats:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-blue-600 to-blue-800 text-white rounded-lg p-6 shadow-lg">
        <h1 className="text-3xl font-bold mb-2">
          Premier League Match Predictor
        </h1>
        <p className="text-blue-100">
          AI-powered predictions using Logistic Regression
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow p-6 border-l-4 border-blue-500">
          <div className="text-sm font-medium text-gray-500 uppercase">
            Current Accuracy
          </div>
          <div className="text-4xl font-bold text-gray-900 mt-2">
            {loading ? '...' : `${stats.accuracy.toFixed(1)}%`}
          </div>
          <div className="text-sm text-gray-600 mt-1">
            {stats.correct} / {stats.total} predictions
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6 border-l-4 border-green-500">
          <div className="text-sm font-medium text-gray-500 uppercase">
            Best Model
          </div>
          <div className="text-2xl font-bold text-gray-900 mt-2">
            Logistic Regression
          </div>
          <div className="text-sm text-gray-600 mt-1">
            Training: {models.logistic_regression ? `${(models.logistic_regression * 100).toFixed(1)}%` : '...'}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6 border-l-4 border-purple-500">
          <div className="text-sm font-medium text-gray-500 uppercase">
            Features Used
          </div>
          <div className="text-lg font-semibold text-gray-900 mt-2">
            6 Features
          </div>
          <div className="text-sm text-gray-600 mt-1">
            Form, Goals, Differences
          </div>
        </div>
      </div>

      <GameweekPredictions />
    </div>
  );
};

export default Dashboard;