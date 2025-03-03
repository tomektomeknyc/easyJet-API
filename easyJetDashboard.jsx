import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import axios from "axios";

const EasyJetDashboard = () => {
  const [financialData, setFinancialData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [ratios, setRatios] = useState([]);
  const [newsSentiment, setNewsSentiment] = useState([]);

  useEffect(() => {
    axios.get("/api/financials").then((response) => {
      setFinancialData(response.data);
    });

    axios.get("/api/predictions?days=30").then((response) => {
      setPredictions(response.data);
    });

    axios.get("/api/ratios").then((response) => {
      setRatios(response.data);
    });

    axios.get("/api/news-sentiment").then((response) => {
      setNewsSentiment(response.data);
    });
  }, []);

  return (
    <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
      <Card>
        <CardContent>
          <h2 className="text-xl font-bold">
            Stock Price Prediction (30 Days)
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={predictions}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="predicted_price"
                stroke="#8884d8"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <h2 className="text-xl font-bold">Financial Performance</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={financialData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="revenue" stroke="#82ca9d" />
              <Line type="monotone" dataKey="net_income" stroke="#ff7300" />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <h2 className="text-xl font-bold">Financial Ratios</h2>
          <ul>
            {ratios.map((ratio, index) => (
              <li key={index}>
                {ratio.name}: {ratio.value}
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <h2 className="text-xl font-bold">News Sentiment Analysis</h2>
          <ul>
            {newsSentiment.map((news, index) => (
              <li key={index}>
                {news.date}: {news.sentiment}
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  );
};

export default EasyJetDashboard;
