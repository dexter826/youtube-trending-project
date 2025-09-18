import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Dashboard from "./pages/Dashboard";
import TrendingAnalysis from "./pages/TrendingAnalysis";
import PredictionPage from "./pages/PredictionPage";
import SplashScreen from "./components/SplashScreen";
import { ApiProvider } from "./context/ApiContext";

function App() {
  const [showSplash, setShowSplash] = useState(true);

  const handleSplashComplete = () => {
    setShowSplash(false);
  };

  return (
    <ApiProvider>
      <Router>
        {showSplash ? (
          <SplashScreen onComplete={handleSplashComplete} />
        ) : (
          <div className="min-h-screen bg-gray-50">
            <Navbar />
            <main className="container mx-auto px-4 py-8">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/analysis" element={<TrendingAnalysis />} />
                <Route path="/prediction" element={<PredictionPage />} />
              </Routes>
            </main>
          </div>
        )}
      </Router>
    </ApiProvider>
  );
}

export default App;
