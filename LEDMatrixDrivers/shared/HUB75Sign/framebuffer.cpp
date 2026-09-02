#include "framebuffer.h"

namespace fbuf {

// The single copy of the pixel store. Statically allocated: there is no
// runtime path on which this can fail, which removes an entire class of
// startup error the DMA library used to have.
uint8_t pixels[CANVAS_W * CANVAS_H];

} // namespace fbuf
