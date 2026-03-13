import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

const TeamStats = () => {
  const [standings, setStandings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadStandings();
  }, []);

  const loadStandings = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${api.BASE_URL}/standings`);
      const data = await response.json();
      setStandings(data);
    } catch (err) {
      setError('Failed to load standings');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">Loading standings...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <p className="text-red-800">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          Premier League Standings 2025-26
        </h2>
        <p className="text-gray-600">
          Current league table used for match predictions
        </p>
      </div>

      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Position
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Team
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {standings.map((item, index) => (
              <tr key={index} className={`
                ${item.position <= 4 ? 'bg-green-50' : ''}
                ${item.position >= 18 ? 'bg-red-50' : ''}
                hover:bg-gray-100
              `}>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <span className={`text-sm font-bold ${
                      item.position <= 4 ? 'text-green-600' :
                      item.position >= 18 ? 'text-red-600' :
                      'text-gray-900'
                    }`}>
                      {item.position}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium text-gray-900">
                    {item.team.replace(' FC', '').replace(' AFC', '')}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-bold text-blue-900 mb-2">
          How Table Positions Affect Predictions
        </h3>
        <p className="text-blue-800 text-sm">
          The model uses actual league positions from this table as features in match predictions. 
          Teams higher in the table are given more weight when predicting outcomes.
        </p>
        <div className="mt-4 flex gap-6 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-100 rounded"></div>
            <span className="text-blue-800">Top 4 (Champions League spots)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-100 rounded"></div>
            <span className="text-blue-800">Bottom 3 (Relegation zone)</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TeamStats;
