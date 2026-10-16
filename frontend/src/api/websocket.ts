// src/api/websocket.ts
import { DiagnosisProgress } from '@/types/diagnosis.types';

type WebSocketMessageHandler = (data: DiagnosisProgress) => void;
type WebSocketErrorHandler = (error: Event) => void;

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 3;  // Réduit pour éviter les boucles infinies
  private reconnectDelay = 2000;
  private isConnecting = false;
  private taskId: string | null = null;
  private onMessageHandlers: WebSocketMessageHandler[] = [];
  private onErrorHandlers: WebSocketErrorHandler[] = [];
  private onCloseHandlers: (() => void)[] = [];

  constructor(
    private baseUrl: string = import.meta.env.VITE_WS_URL || 'ws://localhost:8001/ws'
  ) {}

  /**
   * Se connecte au WebSocket pour une tâche donnée
   */
  connect(taskId: string): Promise<void> {
    // NE PAS connecter si taskId est null ou vide
    if (!taskId) {
      console.warn('⚠️ Task ID null ou vide, connexion WebSocket ignorée');
      return Promise.reject(new Error('Task ID invalide'));
    }

    // Si déjà connecté avec le même taskId, ne rien faire
    if (this.ws?.readyState === WebSocket.OPEN && this.taskId === taskId) {
      return Promise.resolve();
    }

    // Si une connexion est en cours, la fermer
    if (this.isConnecting || this.ws) {
      this.disconnect();
    }

    this.taskId = taskId;
    this.isConnecting = true;

    return new Promise((resolve, reject) => {
      try {
        const url = `${this.baseUrl}/${taskId}`;
        console.log(`🔗 Connexion WebSocket à ${url}`);
        this.ws = new WebSocket(url);

        this.ws.onopen = () => {
          console.log(`✅ WebSocket connecté pour la tâche ${taskId}`);
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data) as DiagnosisProgress;
            this.onMessageHandlers.forEach((handler) => handler(data));
          } catch (e) {
            console.error('❌ Erreur de parsing WebSocket:', e);
          }
        };

        this.ws.onerror = (event) => {
          console.error('❌ WebSocket error:', event);
          this.onErrorHandlers.forEach((handler) => handler(event));
          // Ne pas tenter de reconnecter ici, le onclose fera le travail
        };

        this.ws.onclose = (event) => {
          console.warn(`🔌 WebSocket fermé pour la tâche ${taskId} (code: ${event.code})`);
          this.isConnecting = false;
          this.onCloseHandlers.forEach((handler) => handler());
          
          // Si la fermeture n'est pas intentionnelle, tenter de reconnecter
          if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.handleReconnect();
          }
        };
      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  /**
   * Gère la reconnexion automatique
   */
  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('❌ Nombre maximum de reconnexions atteint.');
      return;
    }

    if (!this.taskId) return;

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1);
    console.log(`🔄 Reconnexion dans ${delay}ms (tentative ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    setTimeout(() => {
      if (this.taskId) {
        this.connect(this.taskId).catch(() => {
          // La reconnexion échouera, le handleReconnect sera rappelé
        });
      }
    }, delay);
  }

  /**
   * Ajoute un handler pour les messages
   */
  onMessage(handler: WebSocketMessageHandler): void {
    this.onMessageHandlers.push(handler);
  }

  /**
   * Ajoute un handler pour les erreurs
   */
  onError(handler: WebSocketErrorHandler): void {
    this.onErrorHandlers.push(handler);
  }

  /**
   * Ajoute un handler pour la fermeture
   */
  onClose(handler: () => void): void {
    this.onCloseHandlers.push(handler);
  }

  /**
   * Se déconnecte proprement
   */
  disconnect(): void {
    this.reconnectAttempts = this.maxReconnectAttempts; // Empêche les reconnexions
    if (this.ws) {
      this.ws.close(1000, 'Fermeture volontaire');
      this.ws = null;
    }
    this.isConnecting = false;
    this.taskId = null;
  }

  /**
   * Vérifie si le WebSocket est connecté
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Récupère l'ID de la tâche actuelle
   */
  getTaskId(): string | null {
    return this.taskId;
  }
}

// Instance unique pour toute l'application
export const wsClient = new WebSocketClient();