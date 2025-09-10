import React from 'react';

const WordCloudComponent = ({ words }) => {
  if (!words || words.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Không có dữ liệu đám mây từ khóa
      </div>
    );
  }

  // Simple word cloud using CSS
  const maxValue = Math.max(...words.map(w => w.value));
  
  const getWordStyle = (word) => {
    const fontSize = Math.max(12, Math.min(48, (word.value / maxValue) * 48));
    const colors = [
      '#FF0000', // YouTube Red
      '#FF4444',
      '#FF6666', 
      '#CC0000',
      '#990000',
      '#666666',
      '#888888'
    ];
    const color = colors[Math.floor(Math.random() * colors.length)];
    
    return {
      fontSize: `${fontSize}px`,
      color: color,
      fontWeight: fontSize > 24 ? 'bold' : 'normal',
      margin: '4px 8px',
      display: 'inline-block',
      cursor: 'pointer',
      transition: 'transform 0.2s',
    };
  };

  return (
    <div className="wordcloud-container bg-gray-50 p-4 rounded-lg">
      <div 
        className="flex flex-wrap justify-center items-center min-h-[300px]"
        style={{ lineHeight: '1.2' }}
      >
        {words.slice(0, 30).map((word, index) => (
          <span
            key={index}
            style={getWordStyle(word)}
            className="hover:scale-110 transition-transform"
            title={`"${word.text}" xuất hiện ${word.value} lần`}
            onClick={() => console.log(`Clicked: ${word.text} (${word.value})`)}
          >
            {word.text}
          </span>
        ))}
      </div>
      <div className="mt-4 text-sm text-gray-600">
        <p className="text-center">
          Đám mây từ khóa được tạo từ {words.length} từ phổ biến nhất trong tiêu đề video
        </p>
        <div className="flex justify-center mt-2 space-x-4">
          <span>🔴 Phổ biến nhất</span>
          <span>⚫ Ít phổ biến hơn</span>
        </div>
      </div>
    </div>
  );
};

export default WordCloudComponent;
