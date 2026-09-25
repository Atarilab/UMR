import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const errorBox = document.getElementById('error');

async function loadArray(path, Type) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Failed to load ${path}: HTTP ${response.status}`);
  return new Type(await response.arrayBuffer());
}

function setError(error) {
  errorBox.style.display = 'block';
  errorBox.textContent = error?.stack || String(error);
}

async function main() {
  const manifestResponse = await fetch('./manifest.json');
  if (!manifestResponse.ok) throw new Error(`Failed to load manifest: HTTP ${manifestResponse.status}`);
  const manifest = await manifestResponse.json();
  document.getElementById('title').textContent = manifest.title;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf3f6fa);
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.getElementById('app').appendChild(renderer.domElement);

  const camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 0.005, 20);
  const initialPosition = new THREE.Vector3(0, 0.07, 1.25);
  const initialTarget = new THREE.Vector3(0, 0.07, 0);
  camera.position.copy(initialPosition);
  camera.lookAt(initialTarget);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(initialTarget);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  scene.add(new THREE.AmbientLight(0xffffff, 0.9));
  const light = new THREE.DirectionalLight(0xffffff, 0.8);
  light.position.set(0.3, 0.6, 1.0);
  scene.add(light);

  const lineFinger = document.getElementById('lineFinger');
  manifest.parts.forEach((part, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = part;
    lineFinger.appendChild(option);
  });

  const meshes = [];
  const clouds = [];
  const hands = [];
  for (const hand of manifest.hands) {
    const parts = await loadArray(hand.slot_parts, Uint8Array);
    const handClouds = [];
    for (const sample of hand.samples) {
      const [meshVertices, meshFaces, positions, corr, part, mismatch] = await Promise.all([
        loadArray(sample.mesh_vertices, Float32Array),
        loadArray(sample.mesh_faces, Uint32Array),
        loadArray(sample.positions, Float32Array),
        loadArray(sample.color_correspondence, Uint8Array),
        loadArray(sample.color_part, Uint8Array),
        loadArray(sample.color_mismatch, Uint8Array),
      ]);
      const meshGeometry = new THREE.BufferGeometry();
      meshGeometry.setAttribute('position', new THREE.BufferAttribute(meshVertices, 3));
      meshGeometry.setIndex(new THREE.BufferAttribute(meshFaces, 1));
      meshGeometry.computeVertexNormals();
      const mesh = new THREE.Mesh(
        meshGeometry,
        new THREE.MeshLambertMaterial({ color: 0xc3ccd6, transparent: true, opacity: 0.35, depthWrite: false, side: THREE.DoubleSide }),
      );
      scene.add(mesh);
      meshes.push(mesh);

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.Uint8BufferAttribute(corr.slice(), 3, true));
      const points = new THREE.Points(
        geometry,
        new THREE.PointsMaterial({ size: 0.0035, sizeAttenuation: true, vertexColors: true }),
      );
      points.userData = { colors: { correspondence: corr, part, mismatch }, positions, hand, label: sample.label };
      scene.add(points);
      clouds.push(points);
      handClouds.push(points);
    }
    hands.push({ hand, parts, clouds: handClouds });
  }

  const lineGeometry = new THREE.BufferGeometry();
  const lines = new THREE.LineSegments(lineGeometry, new THREE.LineBasicMaterial({ color: 0x2a3a4a, transparent: true, opacity: 0.35 }));
  scene.add(lines);
  const selectGeometry = new THREE.BufferGeometry();
  const selectLine = new THREE.LineSegments(selectGeometry, new THREE.LineBasicMaterial({ color: 0xe8364a }));
  scene.add(selectLine);
  const markers = new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.PointsMaterial({ size: 0.012, color: 0xe8364a, sizeAttenuation: true, depthTest: false }),
  );
  scene.add(markers);

  function colorMode() {
    return document.querySelector('input[name="colorMode"]:checked').value;
  }
  function updateColors() {
    const mode = colorMode();
    for (const cloud of clouds) {
      cloud.geometry.setAttribute('color', new THREE.Uint8BufferAttribute(cloud.userData.colors[mode].slice(), 3, true));
    }
  }
  function updateLines() {
    const finger = Number(lineFinger.value);
    const segments = [];
    if (finger >= 0) {
      for (const { parts, clouds: pair } of hands) {
        const a = pair[0].userData.positions;
        const b = pair[1].userData.positions;
        for (let i = 0; i < parts.length; i += 1) {
          if (parts[i] !== finger) continue;
          segments.push(a[3 * i], a[3 * i + 1], a[3 * i + 2], b[3 * i], b[3 * i + 1], b[3 * i + 2]);
        }
      }
    }
    lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(segments, 3));
  }
  document.querySelectorAll('input[name="colorMode"]').forEach((input) => input.addEventListener('change', updateColors));
  lineFinger.addEventListener('change', updateLines);
  document.getElementById('showMeshes').addEventListener('change', (event) => { meshes.forEach((m) => { m.visible = event.target.checked; }); });
  document.getElementById('showSlots').addEventListener('change', (event) => { clouds.forEach((c) => { c.visible = event.target.checked; }); });
  document.getElementById('pointSize').addEventListener('input', (event) => {
    clouds.forEach((c) => { c.material.size = Number(event.target.value); });
  });
  document.getElementById('resetCamera').addEventListener('click', () => {
    camera.position.copy(initialPosition);
    controls.target.copy(initialTarget);
    controls.update();
  });

  const raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = 0.004;
  const pointer = new THREE.Vector2();
  renderer.domElement.addEventListener('click', (event) => {
    pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
    pointer.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(clouds.filter((c) => c.visible))[0];
    if (!hit) return;
    const entry = hands.find((h) => h.clouds.includes(hit.object));
    const index = hit.index;
    const a = entry.clouds[0].userData.positions;
    const b = entry.clouds[1].userData.positions;
    const pa = [a[3 * index], a[3 * index + 1], a[3 * index + 2]];
    const pb = [b[3 * index], b[3 * index + 1], b[3 * index + 2]];
    selectGeometry.setAttribute('position', new THREE.Float32BufferAttribute([...pa, ...pb], 3));
    markers.geometry.setAttribute('position', new THREE.Float32BufferAttribute([...pa, ...pb], 3));
    const part = manifest.parts[entry.parts[index]];
    document.getElementById('selection').innerHTML =
      `${entry.hand.side} hand, slot <b>${index}</b>, part <b>${part}</b>`;
  });

  const rows = manifest.hands.map((hand) => {
    const body = manifest.parts.map((part, k) => {
      const s = hand.stats[part];
      const [r, g, b] = manifest.part_colors[k];
      return `<tr><td><span class="swatch" style="background:rgb(${r},${g},${b})"></span>${part}</td>` +
        `<td>${s.slots}</td><td>${(100 * s.mapped_to_same_link_group).toFixed(0)}%</td>` +
        `<td>${s.robot_surface_distance_mm.toFixed(1)}</td></tr>`;
    }).join('');
    return `<b>${hand.side} hand</b><table><tr><th>part</th><th>slots</th><th>same finger</th><th>mm</th></tr>${body}</table>`;
  });
  document.getElementById('stats').innerHTML = rows.join('<br>') +
    '<br>"same finger": robot slot lies on the same robot finger. "mm": robot slot distance to the robot surface.';

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
}

main().catch(setError);
