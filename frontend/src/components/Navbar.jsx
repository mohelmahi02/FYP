import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar = () => {
  const location = useLocation();

  const isActive = (path) => {
    return location.pathname === path;
  };

  const linkClass = (path) => {
    const base = "px-4 py-2 rounded-md font-medium transition-colors";
    return isActive(path)
      ? `${base} bg-blue-700 text-white`
      : `${base} text-blue-100 hover:bg-blue-700 hover:text-white`;
  };

  return (
    <nav className="bg-blue-600 shadow-lg">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="flex items-center">
            <span className="text-white text-xl font-bold">
              ⚽ PL Predictor
            </span>
          </Link>

          <div className="flex space-x-2">
            <Link to="/" className={linkClass('/')}>
              Dashboard
            </Link>
            <Link to="/history" className={linkClass('/history')}>
              History
            </Link>
            <Link to="/models" className={linkClass('/models')}>
              Models
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;