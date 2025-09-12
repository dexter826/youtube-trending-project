import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import TrendingAnalysis from './pages/TrendingAnalysis';
import PredictionPage from './pages/PredictionPage';
import ModelEvaluation from './pages/ModelEvaluation';
import { ApiProvider } from './context/ApiContext';

function App() {
  return (
    <ApiProvider>
      <Router>
        <div className="min-h-screen bg-gray-50">
          <Navbar />
          <main className="container mx-auto px-4 py-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/analysis" element={<TrendingAnalysis />} />
              <Route path="/prediction" element={<PredictionPage />} />
              <Route path="/evaluation" element={<ModelEvaluation />} />
            </Routes>
          </main>
        </div>
      </Router>
    </ApiProvider>
  );
}

export default App;