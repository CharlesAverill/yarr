/**
 * @file
 * @author Charles Averill <charlesaverill>
 * @date   25-Feb-2022
 * @brief Description
*/

#include "input.cuh"

void input_loop(sfRenderWindow *window, sfEvent *event)
{
    while (sfRenderWindow_pollEvent(window, event)) {
        // Close Window
        if ((*event).type == sfEvtClosed) {
            sfRenderWindow_close(window);
        }
    }
}
