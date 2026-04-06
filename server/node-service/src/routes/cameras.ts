import { Router, Request, Response } from 'express';
import { query } from '../config/database';
import { Camera, ApiResponse } from '../types';

const router = Router();

// GET /api/cameras — list all cameras
router.get('/', async (req: Request, res: Response) => {
  try {
    const cameras = await query<Camera[]>('SELECT * FROM cameras ORDER BY name');
    res.json({ success: true, data: cameras } as ApiResponse<Camera[]>);
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to fetch cameras' });
  }
});

// GET /api/cameras/:id — single camera
router.get('/:id', async (req: Request, res: Response) => {
  try {
    const rows = await query<Camera[]>('SELECT * FROM cameras WHERE id = ?', [req.params.id]);
    if (rows.length === 0) {
      return res.status(404).json({ success: false, error: 'Camera not found' });
    }
    res.json({ success: true, data: rows[0] } as ApiResponse<Camera>);
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to fetch camera' });
  }
});

// PUT /api/cameras/:id/status — update camera online/offline/error status
router.put('/:id/status', async (req: Request, res: Response) => {
  const { status } = req.body as { status: Camera['status'] };
  const allowed: Camera['status'][] = ['online', 'offline', 'error'];
  if (!allowed.includes(status)) {
    return res.status(400).json({ success: false, error: 'Invalid status value' });
  }
  try {
    await query('UPDATE cameras SET status = ? WHERE id = ?', [status, req.params.id]);
    res.json({ success: true, message: `Camera ${req.params.id} status set to ${status}` });
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to update camera status' });
  }
});

export default router;
