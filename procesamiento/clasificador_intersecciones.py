#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
clasificador_intersecciones.py

Clasificador geométrico puro para pares de triángulos reportados por Open3D.

Distingue:
- topological_adjacency
- coincident_vertex_contact
- coincident_edge_contact
- coplanar_overlap
- transverse_intersection
- unexplained_coplanar_contact
- numerically_ambiguous

No depende de Open3D.
"""

from __future__ import annotations
import math
import numpy as np

ALLOWED_CONTACTS = {
    "topological_adjacency",
    "coincident_vertex_contact",
    "coincident_edge_contact",
}

BLOCKING_INTERSECTIONS = {
    "coplanar_overlap",
    "transverse_intersection",
    "unexplained_coplanar_contact",
    "numerically_ambiguous",
    "degenerate_ambiguous",
}


def _triangle_normal(T):
    """Calcula la normal no unitaria de un triángulo."""
    return np.cross(T[1] - T[0], T[2] - T[0])


def _shared_positions(A, B, eps):
    """Identifica posiciones geométricas compartidas dentro de la tolerancia."""
    pts = []
    pairs = []
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            if np.linalg.norm(a - b) <= eps:
                pairs.append((i, j))
                p = 0.5 * (a + b)
                if not any(np.linalg.norm(p - q) <= eps for q in pts):
                    pts.append(p)
    return np.asarray(pts, dtype=np.float64), pairs


def _point_in_triangle_3d(p, T, eps):
    """Comprueba pertenencia al triángulo considerando plano y tolerancia."""
    v0 = T[1] - T[0]
    v1 = T[2] - T[0]
    v2 = p - T[0]

    n = np.cross(v0, v1)
    nn = np.linalg.norm(n)
    if nn <= eps:
        return False

    if abs(np.dot(p - T[0], n / nn)) > eps:
        return False

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    den = d00 * d11 - d01 * d01
    if abs(den) <= 1e-30:
        return False

    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1.0 - v - w

    local_scale = max(
        math.sqrt(max(d00, 0.0)),
        math.sqrt(max(d11, 0.0)),
        eps,
    )
    btol = 5.0 * eps / local_scale

    return (
        u >= -btol
        and v >= -btol
        and w >= -btol
        and u <= 1.0 + btol
        and v <= 1.0 + btol
        and w <= 1.0 + btol
    )


def _segment_triangle_intersection(p0, p1, T, eps):
    """
    Möller-Trumbore limitado al segmento [p0,p1].
    """
    direction = p1 - p0
    e1 = T[1] - T[0]
    e2 = T[2] - T[0]

    h = np.cross(direction, e2)
    a = np.dot(e1, h)

    scale = max(
        np.linalg.norm(direction),
        np.linalg.norm(e1),
        np.linalg.norm(e2),
        eps,
    )

    if abs(a) <= 1e-12 * scale * scale:
        return None

    f = 1.0 / a
    s = p0 - T[0]
    u = f * np.dot(s, h)

    btol = 5.0 * eps / scale
    if u < -btol or u > 1.0 + btol:
        return None

    q = np.cross(s, e1)
    v = f * np.dot(direction, q)

    if v < -btol or u + v > 1.0 + btol:
        return None

    t = f * np.dot(e2, q)
    if t < -btol or t > 1.0 + btol:
        return None

    t = min(1.0, max(0.0, float(t)))
    return p0 + t * direction


def _dedupe_points(points, eps):
    """Elimina puntos repetidos dentro de la distancia de tolerancia."""
    out = []
    for p in points:
        p = np.asarray(p, dtype=np.float64)
        if not any(np.linalg.norm(p - q) <= eps for q in out):
            out.append(p)
    return np.asarray(out, dtype=np.float64)


def _project_2d(T, normal):
    """Proyecta sobre el plano que descarta la componente dominante de la normal."""
    drop = int(np.argmax(np.abs(normal)))
    keep = [i for i in range(3) if i != drop]
    return T[:, keep]


def _signed_area_2d(poly):
    """Calcula el área firmada del polígono con la fórmula del cordón."""
    poly = np.asarray(poly, dtype=np.float64)
    if len(poly) < 3:
        return 0.0
    return 0.5 * float(
        np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - poly[:, 1] * np.roll(poly[:, 0], -1))
    )


def _ensure_ccw(poly):
    """Devuelve los vértices del polígono en sentido antihorario."""
    p = np.asarray(poly, dtype=np.float64)
    return p if _signed_area_2d(p) >= 0.0 else p[::-1]


def _line_intersection_2d(S, E, A, B, eps):
    """Intersecta rectas 2D y usa el punto medio del segmento si son casi paralelas."""
    r = E - S
    q = B - A

    den = r[0] * q[1] - r[1] * q[0]
    if abs(den) <= eps:
        return 0.5 * (S + E)

    AS = A - S
    t = (AS[0] * q[1] - AS[1] * q[0]) / den
    return S + t * r


def _convex_polygon_clip(subject, clip, eps):
    """
    Sutherland-Hodgman. Ambos polígonos son convexos (triángulos).
    """
    output = [np.asarray(x, dtype=np.float64) for x in _ensure_ccw(subject)]
    clip = _ensure_ccw(clip)

    def inside(p, a, b):
        return ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= -eps

    for i in range(len(clip)):
        A = clip[i]
        B = clip[(i + 1) % len(clip)]

        inp = output
        output = []
        if not inp:
            break

        S = inp[-1]
        for E in inp:
            Ein = inside(E, A, B)
            Sin = inside(S, A, B)

            if Ein:
                if not Sin:
                    output.append(_line_intersection_2d(S, E, A, B, eps))
                output.append(E)
            elif Sin:
                output.append(_line_intersection_2d(S, E, A, B, eps))

            S = E

    return np.asarray(output, dtype=np.float64)


def _point_segment_distance(p, a, b):
    """Calcula la distancia de un punto al segmento cerrado."""
    ab = b - a
    den = float(np.dot(ab, ab))

    if den <= 1e-30:
        return float(np.linalg.norm(p - a))

    t = float(np.dot(p - a, ab) / den)
    t = min(1.0, max(0.0, t))
    return float(np.linalg.norm(p - (a + t * ab)))


def _within_shared_feature(points, shared, tol):
    """Comprueba si los contactos quedan cerca de la característica compartida."""
    if len(shared) == 0:
        return False

    if len(shared) == 1:
        return all(np.linalg.norm(p - shared[0]) <= tol for p in points)

    for p in points:
        d = min(
            _point_segment_distance(
                p,
                shared[i],
                shared[j],
            )
            for i in range(len(shared))
            for j in range(i + 1, len(shared))
        )
        if d > tol:
            return False

    return True


def classify_triangle_pair(
    triangle_a,
    triangle_b,
    triangle_indices_a=None,
    triangle_indices_b=None,
    geometric_epsilon=1e-5,
    contact_locality_tolerance=1e-3,
):
    """
    Clasifica un par que Open3D ya reportó como intersectante.

    La clasificación NO asume ninguna forma del objeto.
    """
    A = np.asarray(triangle_a, dtype=np.float64)
    B = np.asarray(triangle_b, dtype=np.float64)

    if A.shape != (3, 3) or B.shape != (3, 3):
        raise ValueError("Cada triángulo debe tener forma (3,3).")

    # Si comparten índices topológicos, ese contacto es adyacencia normal.
    if triangle_indices_a is not None and triangle_indices_b is not None:
        if set(map(int, triangle_indices_a)) & set(map(int, triangle_indices_b)):
            return {
                "category": "topological_adjacency",
                "blocking": False,
                "shared_geometric_positions": 0,
                "intersection_points": [],
            }

    scale = max(
        max(np.linalg.norm(A[(i + 1) % 3] - A[i]) for i in range(3)),
        max(np.linalg.norm(B[(i + 1) % 3] - B[i]) for i in range(3)),
        geometric_epsilon,
    )

    nA = _triangle_normal(A)
    nB = _triangle_normal(B)
    lenA = float(np.linalg.norm(nA))
    lenB = float(np.linalg.norm(nB))

    if lenA <= geometric_epsilon * scale or lenB <= geometric_epsilon * scale:
        return {
            "category": "degenerate_ambiguous",
            "blocking": True,
            "shared_geometric_positions": 0,
            "intersection_points": [],
        }

    uA = nA / lenA
    uB = nB / lenB

    shared, shared_pairs = _shared_positions(
        A,
        B,
        geometric_epsilon,
    )

    parallel = abs(float(np.dot(uA, uB))) >= 1.0 - 1e-8
    plane_distance = max(abs(float(np.dot(p - A[0], uA))) for p in B)

    # ------------------------------------------------------------------
    # Caso coplanar
    # ------------------------------------------------------------------
    if parallel and plane_distance <= max(geometric_epsilon, 1e-8 * scale):
        A2 = _project_2d(A, uA)
        B2 = _project_2d(B, uA)

        poly = _convex_polygon_clip(
            A2,
            B2,
            eps=max(geometric_epsilon, 1e-12),
        )

        overlap_area = abs(_signed_area_2d(poly)) if len(poly) >= 3 else 0.0

        area_tol = max(
            10.0 * geometric_epsilon * scale,
            100.0 * geometric_epsilon * geometric_epsilon,
        )

        if overlap_area > area_tol:
            return {
                "category": "coplanar_overlap",
                "blocking": True,
                "shared_geometric_positions": int(len(shared)),
                "intersection_points": [],
                "coplanar_overlap_area_projected": float(overlap_area),
            }

        if len(shared) >= 2:
            return {
                "category": "coincident_edge_contact",
                "blocking": False,
                "shared_geometric_positions": int(len(shared)),
                "intersection_points": [],
                "coplanar_overlap_area_projected": float(overlap_area),
            }

        if len(shared) == 1:
            return {
                "category": "coincident_vertex_contact",
                "blocking": False,
                "shared_geometric_positions": 1,
                "intersection_points": [],
                "coplanar_overlap_area_projected": float(overlap_area),
            }

        return {
            "category": "unexplained_coplanar_contact",
            "blocking": True,
            "shared_geometric_positions": 0,
            "intersection_points": [],
            "coplanar_overlap_area_projected": float(overlap_area),
        }

    # ------------------------------------------------------------------
    # Caso no coplanar
    # ------------------------------------------------------------------
    points = []

    for i in range(3):
        p = _segment_triangle_intersection(
            A[i],
            A[(i + 1) % 3],
            B,
            geometric_epsilon,
        )
        if p is not None:
            points.append(p)

        p = _segment_triangle_intersection(
            B[i],
            B[(i + 1) % 3],
            A,
            geometric_epsilon,
        )
        if p is not None:
            points.append(p)

    for p in A:
        if _point_in_triangle_3d(
            p,
            B,
            geometric_epsilon,
        ):
            points.append(p)

    for p in B:
        if _point_in_triangle_3d(
            p,
            A,
            geometric_epsilon,
        ):
            points.append(p)

    points = _dedupe_points(
        points,
        max(5.0 * geometric_epsilon, 1e-12),
    )

    if len(points) == 0:
        return {
            "category": "numerically_ambiguous",
            "blocking": True,
            "shared_geometric_positions": int(len(shared)),
            "intersection_points": [],
        }

    if len(shared) > 0 and _within_shared_feature(
        points,
        shared,
        contact_locality_tolerance,
    ):
        category = "coincident_vertex_contact" if len(shared) == 1 else "coincident_edge_contact"
        return {
            "category": category,
            "blocking": False,
            "shared_geometric_positions": int(len(shared)),
            "intersection_points": points.astype(float).tolist(),
        }

    return {
        "category": "transverse_intersection",
        "blocking": True,
        "shared_geometric_positions": int(len(shared)),
        "intersection_points": points.astype(float).tolist(),
    }
