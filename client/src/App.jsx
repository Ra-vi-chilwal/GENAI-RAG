import { useState, useEffect, useRef } from 'react';
import './App.css';
import axios from 'axios';
function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);
  //sdjasss
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  //sd
  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMsg = { id: Date.now(), sender: 'user', text: input };
    setMessages(m => [...m, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await axios.post(
        `${import.meta.env.VITE_API_URL}/chat`,
        { question: userMsg.text },
        {
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );
      console.log(res);
      // if (!res.ok) throw new Error('network');
      const data = res.data;
      console.log(data, res);
      const botMsg = { id: Date.now() + 1, sender: 'bot', text: data.answer ?? JSON.stringify(data) };
      setMessages(m => [...m, botMsg]);
    } catch (e) {
      console.error(e);
      const errMsg = { id: Date.now() + 2, sender: 'bot', text: 'Error: could not reach server.' };
      setMessages(m => [...m, errMsg]);
    } finally {
      setLoading(false);
    }
  };

  const onKey = (e) => { if (e.key === 'Enter') sendMessage(); };

  return (
    <div className="rag-app">
      <div className="chat-shell">
        <header className="chat-header">
          <div className="logo">RAG Chatbot</div>
          <div className="sub">Retrieval augmented conversational UI</div>
        </header>

        <main className="chat-main">
          <div className="messages">
            {messages.map(m => (
              <div key={m.id} className={`msg-row ${m.sender}`}>
                <div className="bubble">{m.text}</div>
              </div>
            ))}
            {loading && (
              <div className="msg-row bot"><div className="bubble">Typing…</div></div>
            )}
            <div ref={endRef} />
          </div>
        </main>

        <footer className="chat-input">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={onKey}
            placeholder="Ask something about your documents..."
            disabled={loading}
          />
          <button onClick={sendMessage} disabled={loading || !input.trim()}>Send</button>
        </footer>
      </div>
    </div>
  );
}

export default App;
