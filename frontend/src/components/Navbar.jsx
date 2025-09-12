import React, { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  BarChart3,
  TrendingUp,
  Brain,
  Settings,
  Menu,
  X,
  Youtube,
} from "lucide-react";
import { useApi } from "../context/ApiContext";

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();
  const { apiHealth } = useApi();

  const navigation = [
    { name: "Dashboard", href: "/", icon: BarChart3 },
    { name: "Phân tích Trending", href: "/analysis", icon: TrendingUp },
    { name: "Dự đoán Video", href: "/prediction", icon: Brain },
    { name: "Đánh giá Mô hình", href: "/evaluation", icon: Settings },
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <nav className="bg-white shadow-lg border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Logo and brand */}
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-br from-youtube-red to-red-600 rounded-lg">
                <Youtube className="w-6 h-6 text-white" />
              </div>
              <div className="hidden sm:block">
                <h1 className="text-xl font-bold text-red-600">
                  YouTube Analytics
                </h1>
                <p className="text-xs text-gray-500">Big Data & ML Platform</p>
              </div>
            </Link>
          </div>

          {/* Desktop navigation */}
          <div className="hidden md:flex items-center space-x-1">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200 ${
                    isActive(item.href)
                      ? "bg-primary-100 text-primary-700 border border-primary-200"
                      : "text-gray-600 hover:text-gray-900 hover:bg-gray-100"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </div>

          {/* API Status and Mobile menu button */}
          <div className="flex items-center space-x-4">
            {/* API Status Indicator */}
            <div className="hidden sm:flex items-center space-x-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  apiHealth?.status === "healthy"
                    ? "bg-green-500"
                    : "bg-red-500"
                }`}
              />
              <span className="text-xs text-gray-500">
                API {apiHealth?.status === "healthy" ? "Online" : "Offline"}
              </span>
            </div>

            {/* Mobile menu button */}
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="md:hidden inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary-500"
            >
              {isOpen ? (
                <X className="block h-6 w-6" />
              ) : (
                <Menu className="block h-6 w-6" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {isOpen && (
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-gray-50 border-t border-gray-200">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={() => setIsOpen(false)}
                  className={`flex items-center space-x-3 px-3 py-2 rounded-md text-base font-medium transition-colors duration-200 ${
                    isActive(item.href)
                      ? "bg-primary-100 text-primary-700"
                      : "text-gray-600 hover:text-gray-900 hover:bg-gray-100"
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{item.name}</span>
                </Link>
              );
            })}

            {/* Mobile API Status */}
            <div className="flex items-center space-x-3 px-3 py-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  apiHealth?.status === "healthy"
                    ? "bg-green-500"
                    : "bg-red-500"
                }`}
              />
              <span className="text-sm text-gray-500">
                API Status:{" "}
                {apiHealth?.status === "healthy" ? "Online" : "Offline"}
              </span>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
