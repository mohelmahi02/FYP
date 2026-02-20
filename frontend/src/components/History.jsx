import React, { useState, useEffect } from 'react';
import PredictionCard from './PredictionCard';
import { api } from '../services/api';

const History = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      setLoading(true);
      const data = await api.getHistory(30);
      const evaluated = data.predictions.filter(p => p.actual_result !== null);
      setHistory(evaluated);
      setError(null);
    } catch (err) {
      setError('Failed to load history');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">Loading history...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-600">{error}</p>
      </div>
    );
  }

  const correctPredictions = history.filter(h => h.correct).length;
  const accuracy = history.length > 0 ? (correctPredictions / history.length * 100).toFixed(1) : 0;

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">
          Prediction History
        </h2>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-3xl font-bold text-blue-600">{history.length}</div>
            <div className="text-sm text-gray-500">Total Predictions</div>
          </div>
          <div>
            <div className="text-3xl font-bold text-green-600">{correctPredictions}</div>
            <div className="text-sm text-gray-500">Correct</div>
          </div>
          <div>
            <div className="text-3xl font-bold text-purple-600">{accuracy}%</div>
            <div className="text-sm text-gray-500">Accuracy</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {history.map((pred, index) => (
          <PredictionCard
            key={index}
            homeTeam={pred.home_team}
            awayTeam={pred.away_team}
            prediction={pred.prediction}
            homeWinProb={pred.home_win_prob}
            drawProb={pred.draw_prob}
            awayWinProb={pred.away_win_prob}
            actualResult={pred.actual_result}
            correct={pred.correct}
            homeGoals={pred.home_goals}
            awayGoals={pred.away_goals}
          />
        ))}
      </div>

      {history.length === 0 && (
        <div className="text-center text-gray-500 py-12">
          No prediction history available
        </div>
      )}
    </div>
  );
};

export default History;