import React from 'react';

const PredictionCard = ({ 
  homeTeam, 
  awayTeam, 
  prediction, 
  homeWinProb, 
  drawProb, 
  awayWinProb,
  actualResult,
  correct,
  homeGoals,
  awayGoals
}) => {
  const getCardStyle = () => {
    if (actualResult === null || actualResult === undefined) {
      return 'bg-white border-gray-200';
    }
    return correct ? 'bg-green-50 border-green-300' : 'bg-red-50 border-red-300';
  };

  const getResultIcon = () => {
    if (actualResult === null || actualResult === undefined) return null;
    return correct ? '✓' : '✗';
  };

  return (
    <div className={`border-2 rounded-lg p-4 shadow-sm ${getCardStyle()}`}>
      <div className="flex justify-between items-start mb-3">
        <div className="flex-1">
          <div className="text-lg font-semibold flex items-center justify-between">
            <span>{homeTeam}</span>
            {actualResult && homeGoals !== null && (
              <span className="text-2xl font-bold mx-4">{homeGoals}</span>
            )}
          </div>
          <div className="text-lg font-semibold flex items-center justify-between mt-1">
            <span>{awayTeam}</span>
            {actualResult && awayGoals !== null && (
              <span className="text-2xl font-bold mx-4">{awayGoals}</span>
            )}
          </div>
        </div>
        {getResultIcon() && (
          <span className={`text-3xl font-bold ${correct ? 'text-green-600' : 'text-red-600'}`}>
            {getResultIcon()}
          </span>
        )}
      </div>

      <div className="mt-3 pt-3 border-t border-gray-200">
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium text-gray-600">Prediction:</span>
          <span className="font-bold text-blue-600">{prediction}</span>
        </div>

        <div className="mt-2 space-y-1">
          <div className="flex items-center text-xs">
            <span className="w-16">Home</span>
            <div className="flex-1 bg-gray-200 rounded-full h-2 mx-2">
              <div 
                className="bg-blue-500 h-2 rounded-full" 
                style={{ width: `${homeWinProb * 100}%` }}
              />
            </div>
            <span className="w-10 text-right">{(homeWinProb * 100).toFixed(1)}%</span>
          </div>

          <div className="flex items-center text-xs">
            <span className="w-16">Draw</span>
            <div className="flex-1 bg-gray-200 rounded-full h-2 mx-2">
              <div 
                className="bg-gray-500 h-2 rounded-full" 
                style={{ width: `${drawProb * 100}%` }}
              />
            </div>
            <span className="w-10 text-right">{(drawProb * 100).toFixed(1)}%</span>
          </div>

          <div className="flex items-center text-xs">
            <span className="w-16">Away</span>
            <div className="flex-1 bg-gray-200 rounded-full h-2 mx-2">
              <div 
                className="bg-red-500 h-2 rounded-full" 
                style={{ width: `${awayWinProb * 100}%` }}
              />
            </div>
            <span className="w-10 text-right">{(awayWinProb * 100).toFixed(1)}%</span>
          </div>
        </div>

        {actualResult && (
          <div className="mt-2 pt-2 border-t border-gray-200 text-sm">
            <span className="text-gray-600">Actual: </span>
            <span className="font-semibold">{actualResult}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionCard;