import { useEffect, useRef, useState } from "react";
import { WS_URL } from "./api";

/**
 * WebSocket live feed with exponential back-off reconnection and a heartbeat.
 * Returns the connection status: "connecting" | "open" | "reconnecting".
 * `onEvent` receives parsed JSON messages; a synthetic {type:"reconnected"} is delivered after
 * a re-connection so the caller can re-sync anything missed while offline.
 */
export function useLiveFeed(onEvent) {
  const [status, setStatus] = useState("connecting");
  const handlerRef = useRef(onEvent);

  useEffect(() => {
    handlerRef.current = onEvent;
  }, [onEvent]);

  useEffect(() => {
    let ws = null;
    let closedByUs = false;
    let attempt = 0;
    let everConnected = false;
    let retryTimer = null;
    let pingTimer = null;

    const connect = () => {
      setStatus(everConnected ? "reconnecting" : "connecting");
      try {
        ws = new WebSocket(WS_URL);
      } catch (e) {
        scheduleRetry();
        return;
      }
      ws.onopen = () => {
        const wasReconnect = everConnected;
        everConnected = true;
        attempt = 0;
        setStatus("open");
        if (wasReconnect) handlerRef.current({ type: "reconnected" });
        pingTimer = setInterval(() => {
          if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "ping" }));
        }, 20000);
      };
      ws.onmessage = (e) => {
        try {
          handlerRef.current(JSON.parse(e.data));
        } catch (err) {
          console.warn("Ignoring malformed WebSocket message", err);
        }
      };
      ws.onerror = () => {
        try {
          ws.close();
        } catch (e) {
          /* already closed */
        }
      };
      ws.onclose = () => {
        clearInterval(pingTimer);
        if (!closedByUs) {
          setStatus("reconnecting");
          scheduleRetry();
        }
      };
    };

    const scheduleRetry = () => {
      const delay = Math.min(15000, 1000 * 2 ** attempt) + Math.random() * 500;
      attempt += 1;
      retryTimer = setTimeout(connect, delay);
    };

    connect();
    return () => {
      closedByUs = true;
      clearTimeout(retryTimer);
      clearInterval(pingTimer);
      if (ws) ws.close();
    };
  }, []);

  return status;
}
