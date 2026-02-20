import React, { useState, useEffect } from 'react';
import PredictionCard from './PredictionCard';
import { api } from '../services/api';

const GameweekPredictions = () => {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadPredictions();
  }, []);

  const loadPredictions = async () => {
    try {
      setLoading(true);
      const data = await api.getPredictions();
      setPredictions(data.predictions || []);
      setError(null);
    } catch (err) {
      setError('Failed to load predictions');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">Loading predictions...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-600">{error}</p>
        <button 
          onClick={loadPredictions}
          className="mt-2 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">
          Upcoming Gameweek Predictions
        </h2>
        <span className="text-sm text-gray-500">
          {predictions.length} matches
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {predictions.map((pred, index) => (
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

      {predictions.length === 0 && (
        <div className="text-center text-gray-500 py-12">
          No predictions available
        </div>
      )}
    </div>
  );
};

export default GameweekPredictions;