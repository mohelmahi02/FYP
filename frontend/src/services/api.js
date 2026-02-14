import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

export const api = {
  // Get upcoming predictions
  getPredictions: async () => {
    const response = await axios.get(`${API_BASE_URL}/predictions?limit=10`);
    return response.data;
  },

  // Get prediction history
  getHistory: async (limit = 20) => {
    const response = await axios.get(`${API_BASE_URL}/history?limit=${limit}`);
    return response.data;
  },

  // Get model comparison stats
  getModels: async () => {
    const response = await axios.get(`${API_BASE_URL}/models`);
    return response.data;
  },

  // Get current accuracy
  getAccuracy: async () => {
    const response = await axios.get(`${API_BASE_URL}/accuracy`);
    return response.data;
  },

  // Health check
  healthCheck: async () => {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return response.data;
  }
};