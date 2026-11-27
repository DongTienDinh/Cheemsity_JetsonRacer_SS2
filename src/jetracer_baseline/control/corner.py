# -*- coding: utf-8 -*-
"""Nhan dien doan thang / khuc cua roi chon ga va goc lai tuong ung.

VAN DE THUC TE tren sa ban: cac khuc cua RAT HEP. Bam lane bang PID thuan tuy
phan ung SAU khi da lech - den luc CTE du lon de PID danh het lai thi xe da cat
qua vach roi. Doan thang thi nguoc lai: PID giu xe giua vach nhung ga van bi
gioi han boi mot `v_max` chung, nen mat thoi gian o cho de an diem nhat.

GIAI PHAP - hai phan tach roi:

1. HAI MUC GA thay vi mot. `v_straight` cho doan thang, `v_corner` cho khuc cua,
   chuyen giua hai muc theo do cong CO TRE (hysteresis). Khong dung mot nguong
   duy nhat: do cong dao quanh nguong se lam ga bat/tat lien tuc, xe giat.

2. LAI FEED-FORWARD theo do cong, cong THANG vao dau ra PID:

       steer = PID(sai_so) + k_ff * do_cong

   PID chi sua cai da sai; feed-forward danh lai NGAY khi nhin thay duong cong,
   khong doi CTE tang. Tren cua hep day la khac biet giua bam duoc va cat cua.
   Trong che do CUA con nhan them `corner_steer_gain` de dat gan het lai.

Do cong den tu da thuc fit cua LaneDetector va da duoc gioi han trong vung thuc
su nhin thay vach, nen no BAO TRUOC khuc cua - ga giam truoc khi xe toi cua chu
khong phai phanh sau khi da lech.

Ga con duoc gioi han toc do doi: len tu tu, xuong nhanh. Bo ga phai duoc ngay
khi thay cua; len ga thi tu tu de banh khong truot.
"""

STRAIGHT = 'THANG'
CORNER = 'CUA'


class CornerController(object):
    def __init__(self, cfg):
        self.reset_from_config(cfg)
        self.reset()

    def reset_from_config(self, cfg):
        c = cfg.get
        # Hai muc ga. `v_straight` mac dinh lay `v_max` de config cu van chay.
        self.v_straight = float(c('control.v_straight', c('control.v_max', 0.20)))
        self.v_corner = float(c('control.v_corner', c('control.v_min', 0.10)))
        self.v_min = float(c('control.v_min', 0.10))
        self.v_max = float(c('control.v_max', 0.20))

        # Hai nguong khac nhau = hysteresis. enter > exit, neu khong se rung.
        self.curve_enter = float(c('control.curve_enter', 0.22))
        self.curve_exit = float(c('control.curve_exit', 0.14))
        if self.curve_exit >= self.curve_enter:
            # Cau hinh sai se lam mat hoan toan tac dung chong rung -> keo ve
            # mot khoang tre toi thieu thay vi im lang chay sai.
            self.curve_exit = self.curve_enter * 0.7

        # GIU GA CUA THEM MOT LUC sau khi ra khoi cua.
        # Ly do: o khuc chu S, do cong di qua 0 khi doi chieu. Nguong hysteresis
        # thuan tuy theo do cong se tha xe ve che do THANG dung luc do - do tren
        # ho so chu S cua duong dua: ga vot 0.120 -> 0.200 trong 3 frame roi tut
        # lai ngay. Tuc la xe TANG TOC dung luc phai dao lai, cho nen no vang
        # ra ngoai o thuy thu hai.
        # Giu them `corner_hold_s` giay lam hai thuy chu S thanh MOT khuc cua
        # lien, khong con khe ho de ga vot len.
        self.corner_hold_s = float(c('control.corner_hold_s', 0.0))
        # PHANH DON DAU theo do cong NHIN THAY phia truoc. Khac han `slowdown`
        # o duoi: `slowdown` chi bo ga khi xe DA lech (phan ung), con cai nay bo
        # ga ngay khi con duong phia truoc bat dau cong - truoc luc xe lech.
        # Do cong den tu da thuc fit tren toan tam nhin nen no bao truoc.
        # Va no LIEN TUC, khong phai buoc nhay o nguong vao cua: xe ga nho dan
        # khi cua den gan thay vi tut mot phat.
        self.curve_brake = float(c('control.curve_brake', 0.0))
        # Do giam toc truoc cua (0..1): thay vi tru hang so co dinh, giam toc muot
        # tu v_straight ve v_corner/v_min ngay khi do cong / diem ngam bao co cua.
        self.pre_corner_decel = max(0.0, min(1.0, float(c('control.pre_corner_decel', 0.0))))
        self.slowdown = float(c('control.slowdown', 0.12))
        self.curve_feedforward = float(c('control.curve_feedforward', 0.9))
        self.corner_steer_gain = float(c('control.corner_steer_gain', 1.6))
        self.road_turn_steer_gain = float(c('control.road_turn_steer_gain', 1.0))
        self.road_turn_release_s = max(0.0, float(c('control.road_turn_release_s', 0.0)))
        # Do tre khi vao cua: giam ga ngay, nhung tang dan he so lai cua de xe
        # khong cat som vao mep trong chu U.
        self.corner_enter_delay_s = max(0.0, float(c('control.corner_enter_delay_s', 0.0)))
        # Nhan dien vung nhieu cua lien tiep (chu U / chu S): rut ngan diem ngam
        # xa, nha lai nhanh khi dao chieu, giam ga nhe va tang do nhay lai cua.
        self.curvy_zone_enabled = bool(c('control.curvy_zone_enabled', False))
        self.curvy_steer_boost = max(1.0, float(c('control.curvy_steer_boost', 1.20)))
        self.curvy_lookahead_scale = max(0.35, min(1.0, float(c('control.curvy_lookahead_scale', 0.65))))
        self.curvy_speed_scale = max(0.50, min(1.0, float(c('control.curvy_speed_scale', 0.88))))
        self.curvy_window_s = max(1.0, float(c('control.curvy_window_s', 6.0)))
        self.curvy_turn_count = max(1, int(c('control.curvy_turn_count', 2)))
        self.curvy_exit_straight_s = max(0.3, float(c('control.curvy_exit_straight_s', 1.8)))
        self.steer_max = float(c('control.steer_max', 0.60))
        # Tran lai RIENG cho khuc cua. Cua cua sa ban rat hep nen can be gan het
        # lai, nhung dung tran do cho ca doan thang thi xe giat va de vuot lane.
        # Mac dinh bang steer_max -> config cu khong doi hanh vi.
        self.corner_steer_max = float(
            c('control.corner_steer_max', self.steer_max))
        if self.corner_steer_max < self.steer_max:
            self.corner_steer_max = self.steer_max

        # Don vi: lenh ga / giay. Xuong nhanh hon len (xem docstring).
        self.throttle_rise_rate = float(c('control.throttle_rise_rate', 0.8))
        self.throttle_fall_rate = float(c('control.throttle_fall_rate', 3.0))

    def reset(self):
        self.mode = STRAIGHT
        self._throttle = 0.0
        self.steer_wanted = 0.0
        self.steer_limit = self.steer_max
        self._hold = 0.0
        self._exit_steer = 0.0
        self._exit_elapsed = 0.0
        self._elapsed = 0.0
        self._enter_delay_elapsed = 0.0
        self._prev_corner_dir = 0
        self.corner_delay_active = False
        self._turn_history = []
        self._curvy_corner_time = 0.0
        self._straight_streak = 0.0
        self.curvy_zone = False

    def effective_lookahead_weight(self, base_weight):
        """Trong vung nhieu cua chu U/S, keo trong so ngam gan hon de khong cat day cung vao trong."""
        w = float(base_weight)
        if self.curvy_zone_enabled and self.curvy_zone:
            return w * self.curvy_lookahead_scale
        return w

    # ------------------------------------------------------------------ mode
    def update_mode(self, curvature):
        mag = abs(float(curvature))
        enter_thr = (self.curve_enter * 0.80
                     if (self.curvy_zone_enabled and self.curvy_zone)
                     else self.curve_enter)
        if self.mode == STRAIGHT:
            if mag >= enter_thr:
                self.mode = CORNER
        else:
            if mag <= self.curve_exit:
                self.mode = STRAIGHT
        return self.mode

    def _update_curvy_zone(self, mode, turn_dir, entered_turn, reversed_turn, dt):
        if not self.curvy_zone_enabled:
            self.curvy_zone = False
            return
        self._elapsed += dt
        cutoff = self._elapsed - self.curvy_window_s
        self._turn_history = [(t, d) for (t, d) in self._turn_history if t >= cutoff]

        if mode == CORNER:
            self._straight_streak = 0.0
            self._curvy_corner_time = min(4.0, self._curvy_corner_time + dt)
            if (entered_turn or reversed_turn) and turn_dir != 0:
                if (not self._turn_history or
                        self._turn_history[-1][1] != turn_dir or
                        (self._elapsed - self._turn_history[-1][0]) >= 0.35):
                    self._turn_history.append((self._elapsed, turn_dir))
            dirs = set(d for (_t, d) in self._turn_history if d != 0)
            has_reversal = (len(dirs) >= 2)
            if (has_reversal or
                    len(self._turn_history) >= self.curvy_turn_count or
                    self._curvy_corner_time >= 1.6):
                self.curvy_zone = True
        else:
            self._straight_streak += dt
            self._curvy_corner_time = max(0.0, self._curvy_corner_time - 0.5 * dt)
            if self._straight_streak >= self.curvy_exit_straight_s:
                self.curvy_zone = False
                self._turn_history = []

    def _steer_at_road_exit(self, steer, limit, road_turn, lane_found, dt):
        duration = self.road_turn_release_s
        if self.curvy_zone_enabled and self.curvy_zone:
            # Trong vung chu S/U can nha lai cua cu nhanh hon de dao chieu kip.
            duration *= 0.40
        if not lane_found or duration <= 0.0:
            self._exit_steer = 0.0
            return steer
        if road_turn or (self.mode == CORNER and self._exit_steer):
            direction = road_turn or self._exit_steer
            self._exit_steer = max(-limit, min(limit, steer)) if steer * direction > 0.0 else 0.0
            self._exit_elapsed = 0.0
            return steer
        if self._exit_steer == 0.0:
            return steer

        self._exit_elapsed += dt
        if steer * self._exit_steer < 0.0 or self._exit_elapsed >= duration:
            self._exit_steer = 0.0
            return steer
        # Chi giu phan lai dang bi nha. Lenh moi van co the tang lai cung chieu.
        if abs(steer) < abs(self._exit_steer):
            remaining = 1.0 - self._exit_elapsed / duration
            steer += (self._exit_steer - steer) * remaining
        return steer

    # ------------------------------------------------------------------- step
    def step(self, pid_output, cte, curvature, dt, lane_found=True, road_turn=0,
             cte_lookahead=0.0):
        """Tra ve (steer, throttle, mode).

        `pid_output` la dau ra PID da tinh san - CornerController khong so huu
        PID, de vong goi quyet dinh dung sai so nao (cte hay tron voi diem ngam).
        """
        if dt <= 0.0:
            dt = 1e-3
        prev_mode = self.mode
        road_turn = road_turn if lane_found else 0
        if road_turn:
            self.mode = CORNER
        else:
            self.update_mode(curvature)
        mode = self.mode

        # Xac dinh huong cua hien tai de phat hien vao cua moi / dao chieu chu S-U
        curv_f = float(curvature)
        if road_turn:
            turn_dir = 1 if road_turn > 0 else -1
        elif abs(curv_f) >= 0.04:
            turn_dir = 1 if curv_f > 0.0 else -1
        elif abs(pid_output) >= 0.05:
            turn_dir = 1 if pid_output > 0.0 else -1
        else:
            turn_dir = 0

        entered_turn = (prev_mode == STRAIGHT and mode == CORNER)
        reversed_turn = (mode == CORNER and self._prev_corner_dir != 0 and
                         turn_dir != 0 and turn_dir != self._prev_corner_dir)
        self._update_curvy_zone(mode, turn_dir, entered_turn, reversed_turn, dt)

        # --- Do tre vao cua (chi hoan khuếch dai lai, KHONG hoan giam ga!) ---
        self.corner_delay_active = False
        delay_s = self.corner_enter_delay_s
        if self.curvy_zone_enabled and self.curvy_zone and delay_s > 0.0:
            # Trong vung nhieu cua: van giu do tre tranh cat som nhung nhay hon 25%
            delay_s *= 0.75
        if mode == CORNER:
            if entered_turn or reversed_turn:
                self._enter_delay_elapsed = 0.0
            if turn_dir != 0:
                self._prev_corner_dir = turn_dir
            if delay_s > 0.0:
                self._enter_delay_elapsed += dt
                # Bo qua do tre ngay neu xe da lech manh hoac cua qua gat (an toan bien ngoai)
                if abs(float(cte)) >= 0.35 or abs(curv_f) >= 0.50:
                    self._enter_delay_elapsed = delay_s
                delay_alpha = min(1.0, self._enter_delay_elapsed / delay_s)
                self.corner_delay_active = (delay_alpha < 1.0 - 1e-6)
            else:
                delay_alpha = 1.0
        else:
            self._enter_delay_elapsed = 0.0
            self._prev_corner_dir = 0
            delay_alpha = 1.0

        # --- lai: PID + feed-forward theo do cong -------------------------
        ff_scale = (0.35 + 0.65 * delay_alpha) if mode == CORNER else 1.0
        steer = pid_output + self.curve_feedforward * ff_scale * curv_f
        limit = self.steer_max
        if mode == CORNER:
            gain = 1.0 + (self.corner_steer_gain - 1.0) * delay_alpha
            steer *= gain
            if steer * road_turn > 0.0:
                rt_gain = 1.0 + (self.road_turn_steer_gain - 1.0) * delay_alpha
                steer *= rt_gain
            if self.curvy_zone_enabled and self.curvy_zone and self.curvy_steer_boost > 1.0:
                c_boost = 1.0 + (self.curvy_steer_boost - 1.0) * delay_alpha
                steer *= c_boost
            limit = self.corner_steer_max
        steer = self._steer_at_road_exit(steer, limit, road_turn, lane_found, dt)
        self.steer_wanted = steer          # truoc khi cat, de bao chan tran
        steer = max(-limit, min(limit, steer))
        self.steer_limit = limit

        # --- ga: chon muc theo che do, roi giam toc truoc cua ---------------
        if mode == CORNER:
            self._hold = self.corner_hold_s
        else:
            self._hold = max(0.0, self._hold - dt)
        # `_hold > 0` = vua ra khoi cua chua lau -> van giu ga cua. Xem docstring
        # cua `corner_hold_s`: day la thu giu hai thuy chu S lien lai voi nhau.
        in_corner_speed = (mode == CORNER) or (self._hold > 0.0)
        target = self.v_corner if in_corner_speed else self.v_straight
        if self.curvy_zone_enabled and self.curvy_zone:
            target *= self.curvy_speed_scale

        # Do giam toc truoc cua (0..1): ha ga muot tu v_straight xuong v_corner/v_min
        # ngay khi thay do cong hoac diem ngam bao co khuc cua phia truoc.
        fall_rate = self.throttle_fall_rate
        if self.pre_corner_decel > 0.0:
            curv_ratio = min(1.0, abs(curv_f) / max(self.curve_enter, 0.05))
            look_ratio = min(1.0, abs(float(cte_lookahead)) / 0.45)
            approach = 1.0 if (in_corner_speed or road_turn) else max(curv_ratio, 0.85 * look_ratio)
            if approach > 0.15:
                u = (approach - 0.15) / 0.85
                decel_frac = self.pre_corner_decel * u
                target -= (target - self.v_min) * (0.75 * decel_frac)
            fall_rate = self.throttle_fall_rate * (0.40 + 1.60 * self.pre_corner_decel)
        elif self.curve_brake > 0.0:
            target -= self.curve_brake * abs(curv_f)         # tuong thich config cu
        target -= self.slowdown * abs(float(cte))            # phan ung
        if not lane_found:
            # Mat vach thi khong duoc giu ga doan thang: xe dang lai theo gia
            # tri cu, cang nhanh cang xa vach.
            target = min(target, self.v_corner)
        target = max(self.v_min, min(self.v_max, target))

        # --- gioi han toc do doi ga ---------------------------------------
        delta = target - self._throttle
        if delta > 0:
            step = self.throttle_rise_rate * dt
            self._throttle += min(delta, step)
        else:
            step = fall_rate * dt
            self._throttle += max(delta, -step)
        self._throttle = max(0.0, min(self.v_max, self._throttle))

        return steer, self._throttle, mode

    @property
    def throttle(self):
        return self._throttle
