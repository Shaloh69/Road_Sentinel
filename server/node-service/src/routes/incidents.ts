import { Router, Request, Response } from 'express';
import { query } from '../config/database';
import { Incident, ApiResponse } from '../types';
import { io } from '../server';

const router = Router();

// GET /api/incidents?camera_id=&status=&severity=&limit=
router.get('/', async (req: Request, res: Response) => {
  try {
    const { camera_id, status, severity, limit = '50' } = req.query as Record<string, string>;

    let sql = `
      SELECT i.*, c.name AS camera_name, c.location AS camera_location
      FROM incidents i
      JOIN cameras c ON i.camera_id = c.id
      WHERE 1=1
    `;
    const params: (string | number)[] = [];

    if (camera_id) { sql += ' AND i.camera_id = ?'; params.push(camera_id); }
    if (status)    { sql += ' AND i.status = ?';    params.push(status); }
    if (severity)  { sql += ' AND i.severity = ?';  params.push(severity); }

    sql += ' ORDER BY i.timestamp DESC LIMIT ?';
    params.push(Math.min(parseInt(limit), 500));

    const rows = await query<Incident[]>(sql, params);
    res.json({ success: true, data: rows } as ApiResponse<Incident[]>);
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to fetch incidents' });
  }
});

// GET /api/incidents/:id
router.get('/:id', async (req: Request, res: Response) => {
  try {
    const rows = await query<Incident[]>(
      'SELECT * FROM incidents WHERE id = ?',
      [req.params.id]
    );
    if (rows.length === 0) return res.status(404).json({ success: false, error: 'Incident not found' });
    res.json({ success: true, data: rows[0] } as ApiResponse<Incident>);
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to fetch incident' });
  }
});

// POST /api/incidents — create incident and broadcast
router.post('/', async (req: Request, res: Response) => {
  try {
    const inc: Incident = req.body;

    const sql = `
      INSERT INTO incidents
        (camera_id, incident_type, severity, title, description, image_url, video_url, confidence, status, metadata)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;
    const result = await query<{ insertId: number }>(sql, [
      inc.camera_id,
      inc.incident_type,
      inc.severity,
      inc.title,
      inc.description ?? null,
      inc.image_url ?? null,
      inc.video_url ?? null,
      inc.confidence ?? null,
      inc.status ?? 'active',
      inc.metadata ? JSON.stringify(inc.metadata) : null,
    ]);

    const saved: Incident = { ...inc, id: result.insertId, timestamp: new Date() };

    // Broadcast to camera subscribers and a global incidents room
    io.to(`camera:${inc.camera_id}`).emit('incident', { type: 'incident', data: saved });
    io.to('incidents').emit('incident', { type: 'incident', data: saved });

    res.status(201).json({ success: true, data: saved } as ApiResponse<Incident>);
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to save incident' });
  }
});

// PUT /api/incidents/:id/status — resolve / mark false alarm / investigate
router.put('/:id/status', async (req: Request, res: Response) => {
  const { status, resolved_by, notes } = req.body as {
    status: Incident['status'];
    resolved_by?: string;
    notes?: string;
  };
  const allowed: Incident['status'][] = ['active', 'resolved', 'false_alarm', 'investigating'];
  if (!allowed.includes(status)) {
    return res.status(400).json({ success: false, error: 'Invalid status value' });
  }
  try {
    await query(
      `UPDATE incidents SET status = ?, resolved_by = ?, notes = ?,
       resolved_at = IF(? = 'resolved', NOW(), resolved_at)
       WHERE id = ?`,
      [status, resolved_by ?? null, notes ?? null, status, req.params.id]
    );
    res.json({ success: true, message: `Incident ${req.params.id} updated to ${status}` });
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to update incident' });
  }
});

export default router;
