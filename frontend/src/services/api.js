import axios from 'axios';

// Use local backend in development, Render in production
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://fyp-backend-ly40.onrender.com/api'
  : 'http://localhost:5000/api';

export const api = {
  BASE_URL: API_BASE_URL,
  
  getPredictions: async () => {
    const response = await axios.get(`${API_BASE_URL}/predictions?limit=10`);
    return response.data;
  },
  
  getHistory: async (limit = 20) => {
    const response = await axios.get(`${API_BASE_URL}/history?limit=${limit}`);
    return response.data;
  },
  
  getModels: async () => {
    const response = await axios.get(`${API_BASE_URL}/models`);
    return response.data;
  },
  
  getAccuracy: async () => {
    const response = await axios.get(`${API_BASE_URL}/accuracy`);
    return response.data;
  },
  
  healthCheck: async () => {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return response.data;
  }
};