// src/components/NetworkGraph.js
import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';

const NetworkGraph = () => {
  const [graphData, setGraphData] = useState(null);

  useEffect(() => {
    // Generate synthetic data
    const generateGraphData = () => {
      const numSupermarkets = 10;
      const numFoodbanks = 10;
      
      // Generate random positions for nodes
      const positions = {};
      const supermarkets = [];
      const foodbanks = [];
      const edges = [];
      const foodbankSatisfaction = {};

      // Generate positions and nodes
      for (let i = 0; i < numSupermarkets; i++) {
        const id = `S${i}`;
        positions[id] = {
          x: Math.random() * 2 - 1,
          y: Math.random() * 2 - 1
        };
        supermarkets.push(id);
      }

      for (let i = 0; i < numFoodbanks; i++) {
        const id = `F${i}`;
        positions[id] = {
          x: Math.random() * 2 - 1,
          y: Math.random() * 2 - 1
        };
        foodbanks.push(id);
        foodbankSatisfaction[id] = 0;
      }

      // Generate edges
      foodbanks.forEach(fb => {
        let connections = 0;
        supermarkets.forEach(sm => {
          if (Math.random() < 0.3) {
            const weight = Math.floor(Math.random() * 100);
            edges.push({
              from: sm,
              to: fb,
              weight: weight
            });
            foodbankSatisfaction[fb] += weight;
            connections++;
          }
        });
        
        // Ensure at least one connection
        if (connections === 0) {
          const randomSupermarket = supermarkets[Math.floor(Math.random() * numSupermarkets)];
          const weight = Math.floor(Math.random() * 100);
          edges.push({
            from: randomSupermarket,
            to: fb,
            weight: weight
          });
          foodbankSatisfaction[fb] += weight;
        }
      });

      // Calculate satisfaction percentages
      Object.keys(foodbankSatisfaction).forEach(fb => {
        foodbankSatisfaction[fb] = foodbankSatisfaction[fb] / (100 * numSupermarkets);
      });

      return {
        positions,
        supermarkets,
        foodbanks,
        edges,
        foodbankSatisfaction
      };
    };

    setGraphData(generateGraphData());
  }, []);

  if (!graphData) return null;

  const { positions, supermarkets, foodbanks, edges, foodbankSatisfaction } = graphData;

  // Create traces
  const edgeTrace = {
    x: edges.flatMap(edge => [positions[edge.from].x, positions[edge.to].x, null]),
    y: edges.flatMap(edge => [positions[edge.from].y, positions[edge.to].y, null]),
    mode: 'lines',
    line: { color: '#888', width: 0.5 },
    hoverinfo: 'none',
    type: 'scatter'
  };

  const supermarketTrace = {
    x: supermarkets.map(s => positions[s].x),
    y: supermarkets.map(s => positions[s].y),
    mode: 'markers',
    marker: { size: 20, color: 'blue' },
    text: supermarkets,
    hoverinfo: 'text',
    type: 'scatter'
  };

  const foodbankTrace = {
    x: foodbanks.map(f => positions[f].x),
    y: foodbanks.map(f => positions[f].y),
    mode: 'markers',
    marker: {
      size: 20,
      color: foodbanks.map(f => {
        const satisfaction = foodbankSatisfaction[f];
        return `rgb(${Math.floor(255 * (1-satisfaction))},${Math.floor(255 * satisfaction)},0)`;
      })
    },
    text: foodbanks.map(f => `${f} (Fulfillment: ${(foodbankSatisfaction[f] * 100).toFixed(2)}%)`),
    hoverinfo: 'text',
    type: 'scatter'
  };

  return (
    <Plot
      data={[edgeTrace, supermarketTrace, foodbankTrace]}
      layout={{
        title: 'Food Distribution Network',
        showlegend: false,
        hovermode: 'closest',
        margin: { b: 20, l: 5, r: 5, t: 40 },
        xaxis: { showgrid: false, zeroline: false, showticklabels: false },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false },
        width: 600,
        height: 600
      }}
      config={{
        displayModeBar: false // This will remove all those control buttons
      }}
      style={{ 
        backgroundColor: '#F5F5F5', // Background color
        borderRadius: '15px',       // Curved corners
        overflow: 'hidden'         // Ensures the content respects the border radius
      }}
    />
  );
};

export default NetworkGraph;