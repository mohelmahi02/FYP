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
    { name: 'HomePosition', description: 'Form-based ranking (home team)', importance: 'highest' },
    { name: 'AwayPosition', description: 'Form-based ranking (away team)', importance: 'highest' },
    { name: 'AwayDrawRate', description: 'Draw rate in last 5 games (away)', importance: 'high' },
    { name: 'HomeTablePos', description: 'Actual league position (home) ', isNew: true, importance: 'high' },
    { name: 'HomeDrawRate', description: 'Draw rate in last 5 games (home)', importance: 'high' },
    { name: 'PositionGap', description: 'Difference in form rankings', importance: 'medium' },
    { name: 'HomeForm5', description: 'Last 5 games points (home team)', importance: 'medium' },
    { name: 'AwayForm5', description: 'Last 5 games points (away team)', importance: 'medium' },
    { name: 'GoalsCloseness', description: 'Goals difference between teams', importance: 'medium' },
    { name: 'AwayGoalsAvg', description: 'Average goals scored (away)', importance: 'medium' },
    { name: 'AwayConcededAvg', description: 'Average goals conceded (away)', importance: 'low' },
    { name: 'AwayTablePos', description: 'Actual league position (away) ', isNew: true, importance: 'low' },
    { name: 'HomeConcededAvg', description: 'Average goals conceded (home)', importance: 'low' },
    { name: 'FormCloseness', description: 'Form difference between teams', importance: 'low' },
    { name: 'TablePosGap', description: 'Difference in actual positions ', isNew: true, importance: 'low' },
    { name: 'HomeGoalsAvg', description: 'Average goals scored (home)', importance: 'low' }
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
          Training accuracy on historical Premier League data (last 3 seasons: 2023-2026)
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
          Features Used for Prediction (16 Total)
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          Ordered by importance (Logistic Regression model coefficients)
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {features.map((feature, index) => (
            <div key={index} className={`flex items-start space-x-3 p-3 rounded-lg ${
              feature.isNew ? 'bg-green-50 border border-green-200' : 'bg-gray-50'
            }`}>
              <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                feature.importance === 'highest' ? 'bg-red-100' :
                feature.importance === 'high' ? 'bg-orange-100' :
                feature.importance === 'medium' ? 'bg-yellow-100' :
                feature.isNew ? 'bg-green-100' : 'bg-blue-100'
              }`}>
                <span className={`font-bold text-sm ${
                  feature.importance === 'highest' ? 'text-red-600' :
                  feature.importance === 'high' ? 'text-orange-600' :
                  feature.importance === 'medium' ? 'text-yellow-600' :
                  feature.isNew ? 'text-green-600' : 'text-blue-600'
                }`}>{index + 1}</span>
              </div>
              <div>
                <div className="font-semibold text-gray-800">{feature.name}</div>
                <div className="text-sm text-gray-600">{feature.description}</div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 flex gap-4 text-xs text-gray-600">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-100 rounded-full"></div>
            <span>Highest importance</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-orange-100 rounded-full"></div>
            <span>High importance</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-yellow-100 rounded-full"></div>
            <span>Medium importance</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-100 rounded-full"></div>
            <span> New feature</span>
          </div>
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
            <span>Uses actual league positions from API for realistic predictions</span>
          </li>
        </ul>
      </div>
    </div>
  );
};

export default ModelComparison;