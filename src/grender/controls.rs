use crate::glm;

use glm::{Vec3, Mat4, Vec2};
use super::consts::*;

pub struct Camera {
    pub pos: Vec3,
    pub target: Vec3
}

impl Camera {
    pub fn new() -> Self {
        Self { pos: [0.0, 0.0, 3.0].into(), target: [0.0, 0.0, 0.0].into() }
    }

    pub fn get_transform(&self) -> Mat4 {
        glm::look_at(&self.pos, &self.target, &UP)
    }

    pub fn get_quat(&self) -> glm::Quat {
        glm::quat_look_at(&(self.target - self.pos), &UP)
    }

    pub fn relative_pos(&self) -> Vec3 {
        self.pos - self.target
    }

    /// orbit camera around the target on the current ball the camera is on (centered at the
    /// target)
    ///
    /// will move radians equal to the magnitiude of `delta` in the local direction of the
    /// components
    pub fn orbit_target(&mut self, delta: &Vec2) {
        let axis = glm::quat_rotate_vec3(&self.get_quat(), &glm::vec2_to_vec3(&delta)).normalize();
        let quat = glm::quat_rotate_normalized_axis(&self.get_quat(), delta.magnitude(), &axis);
        eprintln!("{}", quat);

        let new_rel_pos = glm::quat_rotate_vec3(&quat, &self.relative_pos());
        self.pos = self.target + new_rel_pos;
        eprintln!("{}", self.pos);
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

pub fn mouse_move(cam: &mut Camera, delta: &(f32, f32)) {
    let vdelta: Vec2 = Into::<Vec2>::into([-delta.1, -delta.0]) * 0.05;
    cam.orbit_target(&vdelta);
}
