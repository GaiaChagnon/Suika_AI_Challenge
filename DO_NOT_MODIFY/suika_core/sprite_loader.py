"""
Sprite Loader
==============

Extracts and manages fruit sprites from sprite sheets.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

SPRITES_DIR = Path(__file__).parent.parent.parent / "assets" / "sprites"

# Sprite positions in fruits.png (x, y, width, height)
# Auto-detected from image analysis
FRUITS_PNG_POSITIONS = {
    "cherry": (144, 138, 96, 108),
    "strawberry": (528, 144, 102, 102),
    "grape": (900, 132, 120, 120),
    "dekopon": (1284, 138, 114, 114),
    "persimmon": (1650, 108, 156, 162),
    "apple": (2034, 108, 144, 156),
    "pear": (2394, 90, 204, 204),
    "melon": (2742, 54, 276, 276),  # The large watermelon
}

# Sprite positions in fruits2.png (honeydew and pineapple)
FRUITS2_PNG_POSITIONS = {
    "honeydew": (48, 78, 240, 216),    # Left: honeydew (greenish melon)
    "pineapple": (468, 36, 216, 300),  # Right: pineapple
}

# Map fruit names to their source files
FRUIT_SOURCES = {
    "cherry": ("fruits.png", FRUITS_PNG_POSITIONS["cherry"]),
    "strawberry": ("fruits.png", FRUITS_PNG_POSITIONS["strawberry"]),
    "grape": ("fruits.png", FRUITS_PNG_POSITIONS["grape"]),
    "dekopon": ("fruits.png", FRUITS_PNG_POSITIONS["dekopon"]),
    "persimmon": ("fruits.png", FRUITS_PNG_POSITIONS["persimmon"]),
    "apple": ("fruits.png", FRUITS_PNG_POSITIONS["apple"]),
    "pear": ("fruits.png", FRUITS_PNG_POSITIONS["pear"]),
    "peach": "standalone",  # Standalone file with white background to remove
    "pineapple": ("fruits2.png", FRUITS2_PNG_POSITIONS["pineapple"]),
    "honeydew": ("fruits2.png", FRUITS2_PNG_POSITIONS["honeydew"]),
    "melon": ("fruits.png", FRUITS_PNG_POSITIONS["melon"]),
}

# Standalone sprite files (need background removal)
STANDALONE_SPRITES = {
    "peach": "Peach.png",
}


class SpriteLoader:
    """
    Loads and caches fruit sprites from sprite sheets.
    
    Handles automatic extraction, scaling, and fallback generation.
    """
    
    def __init__(self, sprites_dir: Optional[Path] = None):
        """
        Initialize sprite loader.
        
        Args:
            sprites_dir: Path to sprites directory. Uses default if None.
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame required for sprite loading")
        
        self._sprites_dir = sprites_dir or SPRITES_DIR
        self._sheets: Dict[str, pygame.Surface] = {}
        self._sprites: Dict[str, pygame.Surface] = {}  # Base sprites at source size
        self._scaled_cache: Dict[Tuple[str, int], pygame.Surface] = {}  # (name, size) -> surface
        
        self._load_sheets()
        self._extract_sprites()
    
    def _load_sheets(self) -> None:
        """Load sprite sheet images."""
        # Try to set up display for image loading
        if pygame.display.get_surface() is None:
            try:
                # Try hidden mode first (pygame 2.0+)
                pygame.display.set_mode((1, 1), pygame.HIDDEN)
            except (pygame.error, AttributeError):
                try:
                    # Fallback: try regular tiny window
                    pygame.display.set_mode((1, 1))
                except pygame.error:
                    # Give up on display, will use fallback sprites
                    pass
        
        for filename in ["fruits.png", "fruits2.png", "ui.png"]:
            path = self._sprites_dir / filename
            if path.exists():
                try:
                    # Try loading as regular image first
                    sheet = pygame.image.load(str(path))
                    if pygame.display.get_surface() is not None:
                        sheet = sheet.convert_alpha()
                    self._sheets[filename] = sheet
                except pygame.error:
                    # Silent fail - will use fallback sprites
                    pass
    
    def _extract_sprites(self) -> None:
        """Extract individual sprites from sheets."""
        for name, source in FRUIT_SOURCES.items():
            if source is None:
                continue
            
            # Handle standalone sprites
            if source == "standalone":
                if name in STANDALONE_SPRITES:
                    sprite = self._load_standalone_sprite(STANDALONE_SPRITES[name])
                    if sprite is not None:
                        self._sprites[name] = sprite
                continue
            
            filename, (x, y, w, h) = source
            if filename not in self._sheets:
                continue
            
            sheet = self._sheets[filename]
            
            # Find actual sprite bounds (non-transparent pixels)
            sprite = self._extract_region(sheet, x, y, w, h)
            if sprite is not None:
                self._sprites[name] = sprite
    
    def _load_standalone_sprite(self, filename: str) -> Optional[pygame.Surface]:
        """
        Load a standalone sprite file, remove white background, and crop to content.
        
        Args:
            filename: Name of the sprite file.
            
        Returns:
            Processed sprite surface with transparent background, cropped to content.
        """
        path = self._sprites_dir / filename
        if not path.exists():
            return None
        
        try:
            img = pygame.image.load(str(path))
            if pygame.display.get_surface() is not None:
                img = img.convert_alpha()
            
            # Remove white background by making it transparent
            img = self._remove_white_background(img)
            
            # Crop to content bounds
            img = self._crop_to_content(img)
            
            return img
        except pygame.error:
            return None
    
    def _crop_to_content(self, surface: pygame.Surface) -> pygame.Surface:
        """
        Crop a surface to its non-transparent content bounds.
        
        Args:
            surface: Input surface with transparent areas.
            
        Returns:
            Cropped surface containing only the visible content.
        """
        width, height = surface.get_size()
        
        # Find bounding box of non-transparent pixels
        min_x, min_y = width, height
        max_x, max_y = 0, 0
        
        for x in range(width):
            for y in range(height):
                pixel = surface.get_at((x, y))
                if len(pixel) > 3 and pixel[3] > 10:  # Has some alpha
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
        
        # If no content found, return original
        if min_x >= max_x or min_y >= max_y:
            return surface
        
        # Add small padding
        padding = 2
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(width - 1, max_x + padding)
        max_y = min(height - 1, max_y + padding)
        
        # Create cropped surface
        content_w = max_x - min_x + 1
        content_h = max_y - min_y + 1
        cropped = pygame.Surface((content_w, content_h), pygame.SRCALPHA)
        cropped.blit(surface, (0, 0), (min_x, min_y, content_w, content_h))
        
        return cropped
    
    def _remove_white_background(self, surface: pygame.Surface) -> pygame.Surface:
        """
        Remove white/light background from a sprite.
        
        Args:
            surface: Input surface.
            
        Returns:
            Surface with white pixels made transparent.
        """
        width, height = surface.get_size()
        result = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Threshold for considering a pixel as "white/background"
        threshold = 240
        
        for x in range(width):
            for y in range(height):
                pixel = surface.get_at((x, y))
                r, g, b = pixel[0], pixel[1], pixel[2]
                
                # Check if it's a white/near-white pixel
                if r > threshold and g > threshold and b > threshold:
                    # Make transparent
                    result.set_at((x, y), (0, 0, 0, 0))
                else:
                    # Keep original pixel
                    alpha = pixel[3] if len(pixel) > 3 else 255
                    result.set_at((x, y), (r, g, b, alpha))
        
        return result
    
    def _extract_region(
        self,
        sheet: pygame.Surface,
        x: int,
        y: int,
        w: int,
        h: int
    ) -> Optional[pygame.Surface]:
        """
        Extract a region from a sprite sheet, auto-cropping to content.
        
        Args:
            sheet: Source sprite sheet.
            x, y, w, h: Region bounds.
            
        Returns:
            Cropped sprite surface, or None if empty.
        """
        # Clamp bounds to sheet size
        sheet_w, sheet_h = sheet.get_size()
        x = max(0, min(x, sheet_w - 1))
        y = max(0, min(y, sheet_h - 1))
        w = min(w, sheet_w - x)
        h = min(h, sheet_h - y)
        
        if w <= 0 or h <= 0:
            return None
        
        # Extract region
        region = pygame.Surface((w, h), pygame.SRCALPHA)
        region.blit(sheet, (0, 0), (x, y, w, h))
        
        return region
    
    def get_sprite(self, name: str, size: int) -> pygame.Surface:
        """
        Get a scaled sprite for a fruit.
        
        Args:
            name: Fruit name.
            size: Desired diameter in pixels.
            
        Returns:
            Scaled sprite surface.
        """
        cache_key = (name, size)
        if cache_key in self._scaled_cache:
            return self._scaled_cache[cache_key]
        
        if name in self._sprites:
            base = self._sprites[name]
            # Scale maintaining aspect ratio
            base_w, base_h = base.get_size()
            scale = size / max(base_w, base_h)
            new_w = int(base_w * scale)
            new_h = int(base_h * scale)
            
            scaled = pygame.transform.smoothscale(base, (new_w, new_h))
            self._scaled_cache[cache_key] = scaled
            return scaled
        
        # No sprite available, create gradient fallback
        return self._create_fallback_sprite(name, size)
    
    def _create_fallback_sprite(self, name: str, size: int) -> pygame.Surface:
        """Create a gradient circle fallback for missing sprites."""
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Define colors for missing fruits
        fallback_colors = {
            "peach": (255, 200, 180),
            "honeydew": (200, 255, 180),
        }
        
        color = fallback_colors.get(name, (200, 200, 200))
        center = size // 2
        
        # Draw gradient circle
        for r in range(center, 0, -1):
            ratio = r / center
            c = tuple(int(c * (0.6 + 0.4 * ratio)) for c in color)
            pygame.draw.circle(surface, c, (center, center), r)
        
        # Add highlight
        highlight_color = tuple(min(255, c + 60) for c in color)
        pygame.draw.circle(
            surface,
            (*highlight_color, 180),
            (center - size // 6, center - size // 6),
            size // 4
        )
        
        # Add simple face
        eye_color = (60, 60, 60)
        eye_size = max(2, size // 12)
        eye_y = center - size // 8
        pygame.draw.circle(surface, eye_color, (center - size // 6, eye_y), eye_size)
        pygame.draw.circle(surface, eye_color, (center + size // 6, eye_y), eye_size)
        
        # Smile
        mouth_y = center + size // 8
        pygame.draw.arc(
            surface,
            eye_color,
            (center - size // 6, mouth_y - size // 12, size // 3, size // 6),
            3.14,
            0,
            max(1, size // 20)
        )
        
        self._scaled_cache[(name, size)] = surface
        return surface
    
    def has_sprite(self, name: str) -> bool:
        """Check if a sprite exists for the given fruit."""
        return name in self._sprites
    
    @property
    def available_sprites(self) -> List[str]:
        """List of fruit names with loaded sprites."""
        return list(self._sprites.keys())


# Global sprite loader instance
_sprite_loader: Optional[SpriteLoader] = None


def get_sprite_loader() -> SpriteLoader:
    """Get or create the global sprite loader."""
    global _sprite_loader
    if _sprite_loader is None:
        pygame.init()  # Ensure pygame is initialized
        _sprite_loader = SpriteLoader()
    return _sprite_loader
