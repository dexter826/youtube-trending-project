import React from 'react';
import { Calendar, Globe, Search, RotateCcw } from 'lucide-react';

const FilterPanel = ({
  countries,
  dates,
  selectedCountry,
  selectedDate,
  onCountryChange,
  onDateChange,
  onAnalyze,
  onReset,
  loading
}) => {
  const handleSubmit = (e) => {
    e.preventDefault();
    onAnalyze();
  };

  const canAnalyze = selectedCountry && selectedDate && !loading;

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-2">
          Tham số phân tích
        </h2>
        <p className="text-gray-600">
          Chọn quốc gia và ngày tháng để phân tích dữ liệu video YouTube thịnh hành
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Country Selection */}
          <div>
            <label htmlFor="country" className="block text-sm font-medium text-gray-700 mb-2">
              <Globe className="inline h-4 w-4 mr-1" />
              Quốc gia
            </label>
            <select
              id="country"
              value={selectedCountry}
              onChange={(e) => onCountryChange(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-youtube-red focus:border-transparent"
              disabled={loading}
            >
              <option value="">Chọn quốc gia...</option>
              {countries.map((country) => (
                <option key={country} value={country}>
                  {country === 'US' && '🇺🇸 Hoa Kỳ'}
                  {country === 'GB' && '🇬🇧 Vương quốc Anh'}
                  {country === 'CA' && '🇨🇦 Canada'}
                  {country === 'DE' && '🇩🇪 Đức'}
                  {country === 'FR' && '🇫🇷 Pháp'}
                  {country === 'JP' && '🇯🇵 Nhật Bản'}
                  {country === 'KR' && '🇰🇷 Hàn Quốc'}
                  {country === 'IN' && '🇮🇳 Ấn Độ'}
                  {country === 'RU' && '🇷🇺 Nga'}
                  {country === 'MX' && '🇲🇽 Mexico'}
                  {!['US', 'GB', 'CA', 'DE', 'FR', 'JP', 'KR', 'IN', 'RU', 'MX'].includes(country) && `🌍 ${country}`}
                </option>
              ))}
            </select>
            {selectedCountry && (
              <p className="mt-1 text-xs text-gray-500">
                {dates.length} ngày có sẵn cho {selectedCountry}
              </p>
            )}
          </div>

          {/* Date Selection */}
          <div>
            <label htmlFor="date" className="block text-sm font-medium text-gray-700 mb-2">
              <Calendar className="inline h-4 w-4 mr-1" />
              Ngày tháng
            </label>
            <select
              id="date"
              value={selectedDate}
              onChange={(e) => onDateChange(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-youtube-red focus:border-transparent"
              disabled={loading || !selectedCountry}
            >
              <option value="">Chọn ngày tháng...</option>
              {dates.map((date) => (
                <option key={date} value={date}>
                  {new Date(date).toLocaleDateString('vi-VN', {
                    weekday: 'long',
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric'
                  })}
                </option>
              ))}
            </select>
            {selectedDate && (
              <p className="mt-1 text-xs text-gray-500">
                Phân tích xu hướng cho {selectedDate}
              </p>
            )}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-3">
          <button
            type="submit"
            disabled={!canAnalyze}
            className={`flex-1 flex items-center justify-center px-6 py-3 rounded-md font-medium transition-colors ${
              canAnalyze
                ? 'bg-youtube-red text-white hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-500'
                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
            }`}
          >
            {loading ? (
              <>
                <div className="loading-spinner w-4 h-4 border-2 border-white border-t-transparent rounded-full mr-2"></div>
                Đang phân tích...
              </>
            ) : (
              <>
                <Search className="h-4 w-4 mr-2" />
                Phân tích xu hướng
              </>
            )}
          </button>

          <button
            type="button"
            onClick={onReset}
            disabled={loading}
            className="px-6 py-3 border border-gray-300 rounded-md font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <RotateCcw className="h-4 w-4 mr-2 inline" />
            Đặt lại
          </button>
        </div>

        {/* Help Text */}
        <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
          <h3 className="text-sm font-medium text-blue-900 mb-1">
            💡 Cách hoạt động
          </h3>
          <p className="text-sm text-blue-700">
            Hệ thống này sử dụng Apache Spark để xử lý dữ liệu YouTube thịnh hành được lưu trữ trong MongoDB. 
            Chọn quốc gia và ngày tháng để xem các video thịnh hành hàng đầu và tạo đám mây từ khóa 
            từ tiêu đề video cho ngày cụ thể đó.
          </p>
        </div>
      </form>
    </div>
  );
};

export default FilterPanel;
