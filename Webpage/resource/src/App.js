// App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Suppliers from './pages/Suppliers';
import Distributors from './pages/Distributors';

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/suppliers" element={<Suppliers />} />
          <Route path="/distributors" element={<Distributors />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;