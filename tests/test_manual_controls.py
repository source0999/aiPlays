from aiplays.cli import _manual_tab_turbo_mapping


def test_manual_tab_mapping_is_restored() -> None:
    import sdl2
    from pyboy.plugins import window_sdl2
    from pyboy.utils import WindowEvent

    tab = sdl2.SDLK_TAB
    before = window_sdl2.KEY_UP.get(tab)
    with _manual_tab_turbo_mapping():
        assert window_sdl2.KEY_UP[tab] is WindowEvent.RELEASE_SPEED_UP
    assert window_sdl2.KEY_UP.get(tab) is before
