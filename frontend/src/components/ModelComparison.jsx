import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

const ModelComparison = () => {
  const [models, setModels] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      setLoading(true);
      const data = await api.getModels();
      setModels(data);
    } catch (err) {
      console.error('Failed to load models:', err);
    } finally {
      setLoading(false);
    }
  };

  const features = [
    { name: 'HomeForm5', description: '5-game form (home team)' },
    { name: 'AwayForm5', description: '5-game form (away team)' },
    { name: 'HomeGoalsAvg', description: 'Average goals scored (home)' },
    { name: 'AwayGoalsAvg', description: 'Average goals scored (away)' },
    { name: 'HomeGoalDiff', description: 'Goal difference (home)' },
    { name: 'AwayGoalDiff', description: 'Goal difference (away)' }
  ];

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">Loading models...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          Model Comparison
        </h2>
        <p className="text-gray-600">
          Training accuracy on historical Premier League data (2023-2024 season)
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow p-6 border-l-4 border-green-500">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-bold text-gray-800">
              Logistic Regression
            </h3>
            <span className="bg-green-100 text-green-800 text-xs font-semibold px-2 py-1 rounded">
              BEST
            </span>
          </div>
          <div className="text-4xl font-bold text-gray-900">
            {models.logistic_regression ? `${(models.logistic_regression * 100).toFixed(2)}%` : '...'}
          </div>
          <p className="text-sm text-gray-600 mt-2">
            Linear model with balanced class weights for fair predictions
          </p>
        </div>

        <div className="bg-white rounded-lg shadow p-6 border-l-4 border-blue-500">
          <h3 className="text-lg font-bold text-gray-800 mb-2">
            Random Forest
          </h3>
          <div className="text-4xl font-bold text-gray-900">
            {models.random_forest ? `${(models.random_forest * 100).toFixed(2)}%` : '...'}
          </div>
          <p className="text-sm text-gray-600 mt-2">
            Ensemble of 200 decision trees voting together
          </p>
        </div>

        <div className="bg-white rounded-lg shadow p-6 border-l-4 border-purple-500">
          <h3 className="text-lg font-bold text-gray-800 mb-2">
            Decision Tree
          </h3>
          <div className="text-4xl font-bold text-gray-900">
            {models.decision_tree ? `${(models.decision_tree * 100).toFixed(2)}%` : '...'}
          </div>
          <p className="text-sm text-gray-600 mt-2">
            Single tree classifier with simple decision rules
          </p>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4">
          Features Used for Prediction
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {features.map((feature, index) => (
            <div key={index} className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                <span className="text-blue-600 font-bold text-sm">{index + 1}</span>
              </div>
              <div>
                <div className="font-semibold text-gray-800">{feature.name}</div>
                <div className="text-sm text-gray-600">{feature.description}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-bold text-blue-900 mb-2">
          Why Logistic Regression?
        </h3>
        <ul className="space-y-2 text-blue-800">
          <li className="flex items-start">
            <span className="mr-2">✓</span>
            <span>Highest accuracy on test data ({models.logistic_regression ? `${(models.logistic_regression * 100).toFixed(1)}%` : '...'})</span>
          </li>
          <li className="flex items-start">
            <span className="mr-2">✓</span>
            <span>Balanced predictions across all three outcomes (Home/Draw/Away)</span>
          </li>
          <li className="flex items-start">
            <span className="mr-2">✓</span>
            <span>Provides probability scores for confidence levels</span>
          </li>
          <li className="flex items-start">
            <span className="mr-2">✓</span>
            <span>Interpretable model suitable for academic dissertation</span>
          </li>
        </ul>
      </div>
    </div>
  );
};

export default ModelComparison;