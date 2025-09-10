import React from 'react';
import { X, AlertTriangle } from 'lucide-react';

const ErrorMessage = ({ message, onDismiss }) => {
  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
      <div className="flex items-start">
        <AlertTriangle className="h-5 w-5 text-red-500 mt-0.5 mr-3 flex-shrink-0" />
        <div className="flex-1">
          <h3 className="text-sm font-medium text-red-800 mb-1">
            Lỗi
          </h3>
          <p className="text-sm text-red-700">
            {message}
          </p>
        </div>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="ml-3 flex-shrink-0 text-red-500 hover:text-red-700 transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>
    </div>
  );
};

export default ErrorMessage;
