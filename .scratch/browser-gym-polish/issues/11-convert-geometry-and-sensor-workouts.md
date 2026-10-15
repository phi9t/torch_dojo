# 11: Convert briefs and drills: geometry and sensor workouts

**What to build:** The browser workouts on geometry and sensors (0062–0066, 0070) get structured briefs with markers and `$…$` math, and two or three drills each, focused on frames, shapes and units. The shared practice scene is named consistently across them.

**Blocked by:** 07

**Status:** done

- [x] Each workout's brief uses Goal / Predict / Watch out / Why where they fit, and names coordinate frames explicitly.
- [x] Each drill's probe runs in the browser kernel in well under a second, and its reasoning explains why, not only what.
- [x] The native `make test` / `make eval` and the catalog build pass for every converted workout.
- [x] e2e smoke: each converted workout opens with typed brief sections and at least one drill that runs.

## Comments

- Shared scene: every Goal is followed by "Practice scene: the parking lot", naming the frames used: world frame (metres, $z$ up, yaw in radians counter-clockwise about $+z$), box-local frame (origin at the box centre, $+x$ along its length, $+y$ left, $+z$ up), ego frame at time $t$ / current ego frame ($+x$ forward, $+y$ left), BEV plane, and voxel grid index $(i, j, k)$. In 0065, $t$ only ever means time; translation is $d$.
- Drills (three each) on frames, shapes and units: 0062 (negative sides multiplying, +1 pixel widths, which axis builds the pair grid), 0063 (`@ R` vs `@ R.T` for row-stored points, degrees vs radians, rotate then translate), 0064 (subtract the centre first, inverse rotation, a point exactly on the face), 0065 (composition order, inverting a rigid pose, masking padded NaNs), 0066 (floor vs truncate for negatives, the half-open far edge, `+=` vs `np.add.at`), 0070 (argmax axis, exact thresholds, two GTs claiming one anchor).
- Native eval verifier counts are unchanged: 6/6, 6/6, 7/7, 5/5, 7/7, 6/6.
- Review fixes: 0066 now teaches bounding the integer voxel index ($0 \le$ index $<$ shape), because floor rounding can map a point just inside the metre bound to index `shape`. Its eval reference has the same weakness; changing verifier semantics was out of scope, so it is left for a follow-up.
- `converted.spec.ts` smoke-tests all eleven converted workouts (tickets 10 and 11).
