import React, { useState, useEffect } from 'react';
import { Toaster } from 'react-hot-toast';
import Header from './components/Header';
import FilterPanel from './components/FilterPanel';
import Dashboard from './components/Dashboard';
import LoadingSpinner from './components/LoadingSpinner';
import ErrorMessage from './components/ErrorMessage';
import { apiService } from './services/apiService';

function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [countries, setCountries] = useState([]);
  const [dates, setDates] = useState([]);
  const [selectedCountry, setSelectedCountry] = useState('');
  const [selectedDate, setSelectedDate] = useState('');
  const [trendingData, setTrendingData] = useState(null);
  const [wordcloudData, setWordcloudData] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);

  // Load initial data on component mount
  useEffect(() => {
    loadInitialData();
  }, []);

  // Load dates when country changes
  useEffect(() => {
    if (selectedCountry) {
      loadDatesForCountry(selectedCountry);
    }
  }, [selectedCountry]);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load health status
      const health = await apiService.getHealth();
      setHealthStatus(health);

      // Load available countries
      const countriesData = await apiService.getCountries();
      setCountries(countriesData.countries || []);

      // Auto-select first country if available
      if (countriesData.countries && countriesData.countries.length > 0) {
        setSelectedCountry(countriesData.countries[0]);
      }

    } catch (err) {
      setError(`Không thể tải dữ liệu ban đầu: ${err.message}`);
      console.error('Error loading initial data:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadDatesForCountry = async (country) => {
    try {
      const datesData = await apiService.getDates(country);
      setDates(datesData.dates || []);
      
      // Auto-select most recent date
      if (datesData.dates && datesData.dates.length > 0) {
        const sortedDates = datesData.dates.sort().reverse();
        setSelectedDate(sortedDates[0]);
      }
    } catch (err) {
      console.error('Error loading dates:', err);
      setDates([]);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedCountry || !selectedDate) {
      setError('Vui lòng chọn cả quốc gia và ngày tháng');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setTrendingData(null);
      setWordcloudData(null);

      // Load trending data and wordcloud data in parallel
      const [trending, wordcloud] = await Promise.all([
        apiService.getTrendingVideos(selectedCountry, selectedDate),
        apiService.getWordcloudData(selectedCountry, selectedDate)
      ]);

      setTrendingData(trending);
      setWordcloudData(wordcloud);

    } catch (err) {
      setError(`Phân tích thất bại: ${err.message}`);
      console.error('Error during analysis:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setTrendingData(null);
    setWordcloudData(null);
    setError(null);
    setSelectedCountry(countries[0] || '');
    setSelectedDate('');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Toaster position="top-right" />
      
      <Header healthStatus={healthStatus} />
      
      <main className="container mx-auto px-4 py-8">
        {/* Filter Panel */}
        <div className="mb-8">
          <FilterPanel
            countries={countries}
            dates={dates}
            selectedCountry={selectedCountry}
            selectedDate={selectedDate}
            onCountryChange={setSelectedCountry}
            onDateChange={setSelectedDate}
            onAnalyze={handleAnalyze}
            onReset={handleReset}
            loading={loading}
          />
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-8">
            <ErrorMessage message={error} onDismiss={() => setError(null)} />
          </div>
        )}

        {/* Loading Spinner */}
        {loading && !trendingData && !wordcloudData && (
          <div className="flex justify-center items-center py-12">
            <LoadingSpinner size="large" message="Đang phân tích dữ liệu YouTube xu hướng..." />
          </div>
        )}

        {/* Dashboard */}
        {(trendingData || wordcloudData) && !loading && (
          <Dashboard
            trendingData={trendingData}
            wordcloudData={wordcloudData}
            selectedCountry={selectedCountry}
            selectedDate={selectedDate}
          />
        )}

        {/* Welcome Message */}
        {!trendingData && !wordcloudData && !loading && !error && (
          <div className="text-center py-16">
            <div className="max-w-2xl mx-auto">
              <h2 className="text-3xl font-bold text-gray-900 mb-4">
                Chào mừng đến với YouTube Trending Analytics
              </h2>
              <p className="text-lg text-gray-600 mb-8">
                Khám phá dữ liệu video YouTube thịnh hành bằng công nghệ Big Data. 
                Chọn quốc gia và ngày tháng để bắt đầu phân tích của bạn.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm text-gray-500">
                <div className="bg-white p-6 rounded-lg shadow-sm">
                  <h3 className="font-semibold text-gray-900 mb-2">📊 Phân tích xu hướng</h3>
                  <p>Xem các video thịnh hành hàng đầu với thống kê và số liệu chi tiết</p>
                </div>
                <div className="bg-white p-6 rounded-lg shadow-sm">
                  <h3 className="font-semibold text-gray-900 mb-2">☁️ Đám mây từ khóa</h3>
                  <p>Hiển thị trực quan các từ phổ biến từ tiêu đề video trong đám mây tương tác</p>
                </div>
                <div className="bg-white p-6 rounded-lg shadow-sm">
                  <h3 className="font-semibold text-gray-900 mb-2">📈 Pipeline Big Data</h3>
                  <p>Được hỗ trợ bởi Apache Spark, MongoDB và backend FastAPI</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="container mx-auto px-4 py-6">
          <div className="text-center text-sm text-gray-500">
            <p>YouTube Trending Analytics - Dự án Big Data</p>
            <p className="mt-1">
              Được xây dựng với Apache Spark, MongoDB, FastAPI và React
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
