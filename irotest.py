import pygame
from pygame.locals import *
import sys, time, random, numpy as np, threading, argparse
from OpenGL.GL import *
from OpenGL.GLU import *
import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None
import torch

# ウィンドウサイズ
WIDTH, HEIGHT = 1280, 720

# 固定フレーム数（ベンチマーク用）
TOTAL_FRAMES = 3000

# 物理パラメータ
GRAVITY = 0.001      # 重力
GROUND_Y = -5.0      # 床面のY座標

# 1粒子あたりのポリゴン数（近似値）
POLYGONS_PER_PARTICLE = 512
# 1フレームあたりの粒子生成数
PARTICLES_PER_FRAME = 300

######################################
# OpenGL 用のライティング初期化
######################################
def initialize_lighting():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    light_position = [10.0, 10.0, 10.0, 1.0]
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

######################################
# テクスチャ読み込み
######################################
def load_texture(filename):
    try:
        texture_surface = pygame.image.load(filename)
    except pygame.error as e:
        print(f"Error loading texture: {e}")
        sys.exit()
    texture_data = pygame.image.tostring(texture_surface, 'RGBA', 1)
    tex_width = texture_surface.get_width()
    tex_height = texture_surface.get_height()
    glEnable(GL_TEXTURE_2D)
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_width, tex_height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return texture_id

######################################
# チューブ描画（流出口）
######################################
def draw_tube():
    glPushMatrix()
    glTranslatef(0.0, 7.0, 0.0)
    glRotatef(90, 1, 0, 0)
    glDisable(GL_TEXTURE_2D)
    glColor3f(0.7, 0.7, 0.7)
    quad = gluNewQuadric()
    gluQuadricNormals(quad, GLU_SMOOTH)
    gluCylinder(quad, 0.5, 0.5, 2.0, 32, 1)
    gluDeleteQuadric(quad)
    glEnable(GL_TEXTURE_2D)
    glPopMatrix()

######################################
# 粒子（ペースト）描画（小さな球体）
######################################
def draw_particle(x, y, z):
    glPushMatrix()
    glTranslatef(x, y, z)
    glColor3f(1.0, 0.8, 0.0)  # オレンジ寄り
    quadric = gluNewQuadric()
    gluSphere(quadric, 0.3, 16, 16)
    gluDeleteQuadric(quadric)
    glPopMatrix()

######################################
# 画面上にテキストを描画する（右上）
######################################
def render_text(text, x, y):
    font = pygame.font.SysFont("Arial", 24)
    text_surface = font.render(text, True, (255, 255, 255))
    text_surface = text_surface.convert_alpha()
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    glWindowPos2i(x, y)
    glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

######################################
# CUDA を用いた重い行列乗算でGPU負荷を上げるワーカー
######################################
def heavy_cuda_worker(stop_event):
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    while not stop_event.is_set():
        try:
            a = torch.randn((6000,6000), device=device)
            b = torch.randn((6000,6000), device=device)
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
        except Exception as e:
            print(e)
        time.sleep(0.05)

######################################
# CUDA を用いた風の影響の計算
######################################
def apply_wind_to_particles(particles, current_time):
    if len(particles) == 0 or not torch.cuda.is_available():
        return
    pos_array = np.array([[p["x"], p["y"], p["z"]] for p in particles], dtype=np.float32)
    pos_tensor = torch.tensor(pos_array, device='cuda')
    wind_x = 0.02 * torch.sin(current_time + pos_tensor[:, 0])
    wind_z = 0.02 * torch.cos(current_time + pos_tensor[:, 2])
    displacement = torch.stack([wind_x, torch.zeros_like(wind_x), wind_z], dim=1)
    new_pos_tensor = pos_tensor + displacement
    new_positions = new_pos_tensor.cpu().numpy()
    for i, p in enumerate(particles):
        p["x"] = new_positions[i, 0]
        p["z"] = new_positions[i, 2]

######################################
# メイン関数（スコア制ベンチマーク）
######################################
def main():
    # コマンドライン引数で実行時間を指定（秒）
    import argparse
    parser = argparse.ArgumentParser(description="3D Paste Benchmark with Score")
    parser.add_argument("--time", type=float, default=60.0,
                        help="Benchmark duration in seconds (default: 60.0)")
    args = parser.parse_args()
    duration_sec = args.time

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D Paste Benchmark (Score-Based)")
    clock = pygame.time.Clock()

    glClearColor(0.2, 0.2, 0.2, 1.0)
    glEnable(GL_DEPTH_TEST)
    initialize_lighting()

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (WIDTH/HEIGHT), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0.0, 0.0, 20.0,  0.0, 0.0, 0.0,  0.0, 1.0, 0.0)

    texture_id = load_texture("mytexture.png")
    if texture_id is None:
        print("Texture load failed.")
        pygame.quit()
        sys.exit()

    # CUDA負荷ワーカー起動
    stop_event = threading.Event()
    cuda_thread = threading.Thread(target=heavy_cuda_worker, args=(stop_event,), daemon=True)
    cuda_thread.start()

    particles = []
    tube_x, tube_y, tube_z = 0.0, 5.0, 0.0

    start_time = time.time()
    frames = 0
    info_update_time = start_time
    info_text = "FPS: 0  CPU: 0%  GPU: 0%"

    while time.time() - start_time < duration_sec:
        for event in pygame.event.get():
            if event.type == QUIT:
                stop_event.set()
                pygame.quit()
                sys.exit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(0.0, 0.0, 20.0,  0.0, 0.0, 0.0,  0.0, 1.0, 0.0)

        draw_tube()

        # 毎フレーム大量の粒子を生成
        for _ in range(PARTICLES_PER_FRAME):
            new_particle = {
                "x": tube_x + random.uniform(-0.5, 0.5),
                "y": tube_y,
                "z": tube_z + random.uniform(-0.5, 0.5),
                "vy": 0.0,
                "vx": random.uniform(-0.005, 0.005),
                "vz": random.uniform(-0.005, 0.005)
            }
            particles.append(new_particle)

        # 粒子更新：重力で下落、床に達したら水平拡散
        for p in particles:
            if p["y"] > GROUND_Y:
                p["vy"] -= GRAVITY
                p["y"] += p["vy"]
            else:
                p["vx"] *= 0.99
                p["vz"] *= 0.99
                p["x"] += p["vx"]
                p["z"] += p["vz"]

        current_time = time.time() - start_time
        apply_wind_to_particles(particles, current_time)

        glBindTexture(GL_TEXTURE_2D, texture_id)
        for p in particles:
            draw_particle(p["x"], p["y"], p["z"])

        # リアルタイム情報更新（0.5秒ごと）
        if time.time() - info_update_time >= 0.5:
            fps = clock.get_fps()
            cpu_usage = psutil.cpu_percent(interval=None)
            if GPUtil:
                gpus = GPUtil.getGPUs()
                gpu_usage = gpus[0].load * 100 if gpus else 0.0
            else:
                gpu_usage = 0.0
            info_text = f"FPS: {fps:.1f}  CPU: {cpu_usage:.1f}%  GPU: {gpu_usage:.1f}%"
            info_update_time = time.time()

        render_text(info_text, WIDTH - 300, HEIGHT - 30)

        pygame.display.flip()
        frames += 1
        clock.tick(0)

    stop_event.set()
    elapsed = time.time() - start_time
    total_polygons = POLYGONS_PER_PARTICLE * PARTICLES_PER_FRAME * frames
    score = total_polygons / elapsed
    print(f"--- BENCHMARK RESULT ---")
    print(f"Total Frames: {frames}")
    print(f"Time Elapsed: {elapsed:.2f} s")
    print(f"Approx. Polygons: {total_polygons}")
    print(f"Score (Polygons/s): {score:.1f}")
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
