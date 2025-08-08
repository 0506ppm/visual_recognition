import React, { useState, useEffect } from 'react';
import './CamWin.css';

function CamWin() {
    const [trans, setTrans] = useState('Processing...');
    
    useEffect(() => {
        const fetchTranslation = async () => {
            try {
                const response = await fetch('http://localhost:5000/getRes', {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    mode: 'cors', // 確保是 cors 模式
                })
                .then(response => response.json())
                .then(data => console.log(data))
                .catch(error => console.error('Error:', error));
                    
                const data = await response.json();
                if (data.msg !== 'Processing') {
                    setTrans(data.msg);
                }
            } catch (error) {
                console.error("Error fetching translation", error);
            }
        };

        const interval = setInterval(fetchTranslation, 3000); // Poll every second

        return () => clearInterval(interval);
    }, []);

    return (
        <div>
            <h1>手語辨識</h1>
            <p>結果: {trans}</p>
            <div className="video-container">
                <img src="http://localhost:5000/video_feed" alt="Video" />
            </div>
        </div>
    );
}

export default CamWin;