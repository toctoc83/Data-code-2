'use client';

import { useMemo, useState } from 'react';

const buttons = [
  ['C', '±', '%', '÷'],
  ['7', '8', '9', '×'],
  ['4', '5', '6', '−'],
  ['1', '2', '3', '+'],
  ['0', '.', '=']
];

function formatNumber(value) {
  if (!Number.isFinite(value)) return 'Fehler';
  return new Intl.NumberFormat('de-DE', {
    maximumFractionDigits: 10
  }).format(value);
}

export default function CalculatorPage() {
  const [display, setDisplay] = useState('0');
  const [storedValue, setStoredValue] = useState(null);
  const [operator, setOperator] = useState(null);
  const [waitingForNext, setWaitingForNext] = useState(false);
  const [history, setHistory] = useState('Bereit');

  const liveExpression = useMemo(() => {
    if (storedValue === null || operator === null) return display;
    return `${formatNumber(storedValue)} ${operator} ${display}`;
  }, [display, storedValue, operator]);

  function inputDigit(digit) {
    if (waitingForNext) {
      setDisplay(digit);
      setWaitingForNext(false);
      return;
    }

    setDisplay((current) => (current === '0' ? digit : current + digit));
  }

  function inputDecimal() {
    if (waitingForNext) {
      setDisplay('0.');
      setWaitingForNext(false);
      return;
    }

    if (!display.includes('.')) {
      setDisplay(display + '.');
    }
  }

  function clearCalculator() {
    setDisplay('0');
    setStoredValue(null);
    setOperator(null);
    setWaitingForNext(false);
    setHistory('Zurückgesetzt');
  }

  function toggleSign() {
    setDisplay((current) => {
      if (current === '0') return current;
      return current.startsWith('-') ? current.slice(1) : `-${current}`;
    });
  }

  function percent() {
    const value = parseFloat(display) / 100;
    setDisplay(String(value));
    setHistory(`${display}% = ${formatNumber(value)}`);
  }

  function calculate(first, second, op) {
    switch (op) {
      case '+':
        return first + second;
      case '−':
        return first - second;
      case '×':
        return first * second;
      case '÷':
        return second === 0 ? NaN : first / second;
      default:
        return second;
    }
  }

  function chooseOperator(nextOperator) {
    const inputValue = parseFloat(display);

    if (storedValue === null) {
      setStoredValue(inputValue);
      setHistory(`${formatNumber(inputValue)} ${nextOperator}`);
    } else if (operator) {
      const result = calculate(storedValue, inputValue, operator);
      setDisplay(String(result));
      setStoredValue(result);
      setHistory(`${formatNumber(storedValue)} ${operator} ${formatNumber(inputValue)} = ${formatNumber(result)}`);
    }

    setOperator(nextOperator);
    setWaitingForNext(true);
  }

  function equals() {
    if (operator === null || storedValue === null) return;

    const inputValue = parseFloat(display);
    const result = calculate(storedValue, inputValue, operator);

    setDisplay(String(result));
    setHistory(`${formatNumber(storedValue)} ${operator} ${formatNumber(inputValue)} = ${formatNumber(result)}`);
    setStoredValue(null);
    setOperator(null);
    setWaitingForNext(true);
  }

  function handleButton(value) {
    if (/^[0-9]$/.test(value)) return inputDigit(value);
    if (value === '.') return inputDecimal();
    if (value === 'C') return clearCalculator();
    if (value === '±') return toggleSign();
    if (value === '%') return percent();
    if (value === '=') return equals();
    return chooseOperator(value);
  }

  return (
    <main className="page-shell">
      <section className="hero-card">
        <div className="title-block">
          <p className="eyebrow">Next.js Calculator</p>
          <h1>Schöner Neon-Taschenrechner</h1>
          <p>
            Ein modernes Mini-Projekt mit React-State, sauberem CSS und einer responsiven Oberfläche.
          </p>
        </div>

        <div className="calculator-card" aria-label="Taschenrechner">
          <div className="display-panel">
            <span className="history">{history}</span>
            <span className="expression">{liveExpression}</span>
            <output className="display">{formatNumber(parseFloat(display))}</output>
          </div>

          <div className="button-grid">
            {buttons.flat().map((button) => (
              <button
                key={button}
                className={`calc-button ${button === '0' ? 'zero' : ''} ${['÷', '×', '−', '+', '='].includes(button) ? 'operator' : ''} ${['C', '±', '%'].includes(button) ? 'utility' : ''}`}
                type="button"
                onClick={() => handleButton(button)}
              >
                {button}
              </button>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
