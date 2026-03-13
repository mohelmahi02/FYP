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
      console.log('Standings data:', data);  
    console.log('First team:', data[0]);   
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
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Pos
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Team
                </th>
                <th className="px-3 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Pl
                </th>
                <th className="px-3 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                  W
                </th>
                <th className="px-3 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                  D
                </th>
                <th className="px-3 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                  L
                </th>
                <th className="px-3 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                  GF
                </th>
                <th className="px-3 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                  GA
                </th>
                <th className="px-3 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                  GD
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Pts
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {standings.map((team, index) => (
                <tr key={index} className={`
                  ${team.position <= 4 ? 'bg-green-50' : ''}
                  ${team.position >= 18 ? 'bg-red-50' : ''}
                  hover:bg-gray-100
                `}>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <span className={`text-sm font-bold ${
                      team.position <= 4 ? 'text-green-600' :
                      team.position >= 18 ? 'text-red-600' :
                      'text-gray-900'
                    }`}>
                      {team.position}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900">
                      {team.team.replace(' FC', '').replace(' AFC', '')}
                    </div>
                  </td>
                  <td className="px-3 py-4 whitespace-nowrap text-center text-sm text-gray-900">
                    {team.played}
                  </td>
                  <td className="px-3 py-4 whitespace-nowrap text-center text-sm text-gray-900">
                    {team.won}
                  </td>
                  <td className="px-3 py-4 whitespace-nowrap text-center text-sm text-gray-900">
                    {team.drawn}
                  </td>
                  <td className="px-3 py-4 whitespace-nowrap text-center text-sm text-gray-900">
                    {team.lost}
                  </td>
                  <td className="px-3 py-4 whitespace-nowrap text-center text-sm text-gray-900">
                    {team.goalsFor}
                  </td>
                  <td className="px-3 py-4 whitespace-nowrap text-center text-sm text-gray-900">
                    {team.goalsAgainst}
                  </td>
                  <td className="px-3 py-4 whitespace-nowrap text-center text-sm">
                    <span className={team.goalDifference >= 0 ? 'text-green-600' : 'text-red-600'}>
                      {team.goalDifference > 0 ? '+' : ''}{team.goalDifference}
                    </span>
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap text-center">
                    <span className="text-sm font-bold text-gray-900">
                      {team.points}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-bold text-blue-900 mb-2">
          Table Key
        </h3>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-blue-800 mb-2">
              <span className="font-semibold">Pl</span> = Played | 
              <span className="font-semibold"> W</span> = Won | 
              <span className="font-semibold"> D</span> = Drawn | 
              <span className="font-semibold"> L</span> = Lost
            </p>
            <p className="text-blue-800">
              <span className="font-semibold">GF</span> = Goals For | 
              <span className="font-semibold"> GA</span> = Goals Against | 
              <span className="font-semibold"> GD</span> = Goal Difference | 
              <span className="font-semibold"> Pts</span> = Points
            </p>
          </div>
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-green-100 rounded"></div>
              <span className="text-blue-800">Top 4 (Champions League)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-red-100 rounded"></div>
              <span className="text-blue-800">Bottom 3 (Relegation)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TeamStats;