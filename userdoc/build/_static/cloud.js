/* ~~~~~~~~~~~~~~
 * cloud.js_t
 * ~~~~~~~~~~~~~~
 *
 * Various bits of javascript driving the moving parts behind various
 * parts of the cloud theme. Handles things such as toggleable sections,
 * collapsing the sidebar, etc.
 *
 * :copyright: Copyright 2011-2012 by Assurance Technologies
 * :license: BSD
 */



  
  


// begin encapsulation
(function (window, $, _){

/* ==========================================================================
 * common helpers
 * ==========================================================================*/
var isUndef = _.isUndefined,
    TEXT_NODE = 3; // could use Node.TEXT_NODE, but IE doesn't define it :(

// helper to generate an absolute url path from a relative one.
// absolute paths passed through unchanged.
// paths treated as relative to <base>,
// if base is omitted, uses directory of current document.
function abspath(path, base) {
    var parts = path.split("/"),
        stack = [];
    if (parts[0]) {
      // if path is relative, put base on stack
      stack = (base || document.location.pathname).split("/");
      // remove blank from leading '/'
      if (!stack[0]) { stack.shift(); }
      // discard filename & blank from trailing '/'
      if (stack.length && !(base && stack[stack.length-1])) { stack.pop(); }
    }
    for (var i=0; i < parts.length; ++i) {
      if (parts[i] && parts[i] != '.') {
        if (parts[i] == '..'){
          stack.pop();
        } else {
          stack.push(parts[i]);
        }
      }
    }
    return "/" + stack.join("/");
}

// helper to normalize urls for comparison
// * strips current document's scheme, host, & path from local document links (just fragment will be left)
// * strips current document's scheme & host from internal urls (just path + fragment will be left)
// * makes all internal url paths absolute
// * external urls returned unchanged.
var hosturl = document.location.href.match(/^[a-z0-9]+:(\/\/)?([^@]*@)?[^/]+/)[0],
    docpath = document.location.pathname;
function shorten_url(url) {
  if (url.indexOf(hosturl) == 0) {
    url = url.substr(hosturl.length) || '/';
  } else if (url && url[0] == '.') {
    url = abspath(url);
  }
  if (url.indexOf(docpath) == 0) {
    url = url.substr(docpath.length);
  }
  if (url == "#") { url = ""; } // border case from sphinxlocaltoc
  return url;
}

// helper that retrieves css property in pixels
function csspx(elem, prop) {
  return 1 * $(elem).css(prop).replace(/px$/, '');
}

/* debugging
window.CloudSphinxTheme = {
  shorten_url: shorten_url,
};
*/

// NOTE: would use $().offset(), but it's document-relative,
//       and we need viewport-relative... which means getBoundingClientRect().
// NOTE: 'window.frameElement' will only work we're embedded in an iframe on same domain.
var parentFrame = window.frameElement;
if(window.parent && window.parent !== window){
    $(window.parent).scroll(function (){ $(window).scroll(); });
}

function leftFrameOffset(){ return parentFrame ? parentFrame.getBoundingClientRect().left : 0; }
function topFrameOffset(){ return parentFrame ? parentFrame.getBoundingClientRect().top : 0; }

function leftViewOffset($node){ return ($node && $node.length > 0) ? $node[0].getBoundingClientRect().left + leftFrameOffset() : 0; } 
function topViewOffset($node){ return ($node && $node.length > 0) ? $node[0].getBoundingClientRect().top + topFrameOffset() : 0; } 

// return normalized nodename, takes in node or jquery selector
// (can't trust nodeName, per http://ejohn.org/blog/nodename-case-sensitivity/)
function nodeName(elem) {
  if (elem && elem.length) { elem = elem[0]; }
  return elem && elem.nodeName.toUpperCase();
}

/* ==========================================================================
 * highlighter #2
 * ==========================================================================
 *
 * Sphinx's highlighter marks some objects when user follows link,
 * but doesn't include section names, etc. This catches those.
 */
$(document).ready(function (){
  // helper to locate highlight target based on #fragment
  function locate_target(){
    // find id referenced by #fragment
    var hash = document.location.hash;
    if(!hash) return null;
    var section = document.getElementById(hash.substr(1));
    if(!section) return null;

    // could be div.section, or hidden span at top of div.section
    var name = nodeName(section);
    if(name != "DIV"){
      if(name == "SPAN" && section.innerHTML == "" &&
         nodeName(section.parentNode) == "DIV")
      {
          section = section.parentNode;
      }
      else if (name == "DT" && section.children.length &&
               $(section).children("tt.descname, code.descname").length > 0)
      {
        // not a section, but an object definition, e.g. a class, func, or attr
        return $(section);
      }
    }
    // now at section div and either way we have to find title element - h2, h3, etc.
    var header = $(section).children("h2, h3, h4, h5, h6").first();
    return header.length ? header : null;
  }

  // init highlight
  var target = locate_target();
  if(target) target.addClass("highlighted");

  // update highlight if hash changes
  $(window).bind("hashchange", function () {
    if(target) target.removeClass("highlighted");
    target = locate_target();
    if(target) target.addClass("highlighted");
  });
});

/* ==========================================================================
 * toggleable sections
 * ==========================================================================
 *
 * Added expand/collapse button to any collapsible RST sections.
 * Looks for sections with CSS class "html-toggle",
 * along with the optional classes "expanded" or "collapsed".
 * Button toggles "html-toggle.expanded/collapsed" classes,
 * and relies on CSS to do the rest of the job displaying them as appropriate.
 */

$(document).ready(function (){
  function init(){
    // get header & section, and add static classes
    var header = $(this);
    var section = header.parent();
    header.addClass("html-toggle-button");

    // helper to test if url hash is within this section
    function contains_hash(){
      var hash = document.location.hash;
      return hash && (section[0].id == hash.substr(1) ||
              section.find(hash.replace(/\./g,"\\.")).length>0);
    }

    // helper to control toggle state
    function set_state(expanded){
      expanded = !!expanded; // toggleClass et al need actual boolean
      section.toggleClass("expanded", expanded);
      section.toggleClass("collapsed", !expanded);
      section.children().toggle(expanded);
      if (!expanded) {
        section.children("span:first-child:empty").show(); /* for :ref: span tag */
        header.show();
      }
    }

    // initialize state
    set_state(section.hasClass("expanded") || contains_hash());

    // bind toggle callback
    header.click(function (evt){
      var state = section.hasClass("expanded")
      if(state && $(evt.target).is(".headerlink")) { return; }
      set_state(!state);
      $(window).trigger('cloud-section-toggled', section[0]);
    });

    // open section if user jumps to it from w/in page
    $(window).bind("hashchange", function () {
      if(contains_hash()) set_state(true);
    });
  }

  $(".html-toggle.section > h2, .html-toggle.section > h3, .html-toggle.section > h4, .html-toggle.section > h5, .html-toggle.section > h6").each(init);
});
/* ==========================================================================
 * collapsible sidebar
 * ==========================================================================
 *
 * Adds button for collapsing & expanding sidebar,
 * which toggles "document.collapsed-sidebar" CSS class,
 * and relies on CSS for actual styling of visible & hidden sidebars.
 */

$(document).ready(function (){
  if(!$('.sphinxsidebar').length){
    return;
  }
  
    var close_arrow = '«';
    var open_arrow = 'sidebar »';
  
  var holder = $('<div class="sidebartoggle"><button id="sidebar-hide" title="click to hide the sidebar">' +
                 close_arrow + '</button><button id="sidebar-show" style="display: none" title="click to show the sidebar">' +
                 open_arrow + '</button></div>');
  var doc = $('div.document');

  var show_btn = $('#sidebar-show', holder);
  var hide_btn = $('#sidebar-hide', holder);
  var copts = { expires: 7, path: abspath(DOCUMENTATION_OPTIONS.URL_ROOT || "") };

  show_btn.click(function (){
    doc.removeClass("collapsed-sidebar");
    hide_btn.show();
    show_btn.hide();
    $.cookie("sidebar", "expanded", copts);
    $(window).trigger("cloud-sidebar-toggled", false);
  });

  hide_btn.click(function (){
    doc.addClass("collapsed-sidebar");
    show_btn.show();
    hide_btn.hide();
    $.cookie("sidebar", "collapsed", copts);
    $(window).trigger("cloud-sidebar-toggled", true);
  });

  var state = $.cookie("sidebar");


  doc.append(holder);

  if (state == "collapsed"){
    doc.addClass("collapsed-sidebar");
    show_btn.show();
    hide_btn.hide();
  }
});
/* ==========================================================================
 * sticky sidebar
 * ==========================================================================
 *
 * Instrument sidebar so that it sticks in place as page is scrolled.
 */
$(document).ready(function (){
  // initialize references to relevant elements
  var holder = $('.document'); // element that sidebar sits within
  var sidebar = $('.sphinxsidebar'); // element we're making "sticky"
  var toc_header = $('.sphinxlocaltoc h3'); // toc header + list control position
      if(!toc_header.length) toc_header = null;
  var toc_list = toc_header ? toc_header.next("ul") : null;
  var toggle = $('.sidebartoggle'); // also make collapse button sticky

  // initialize internal state
  var sticky_disabled = false, // whether sticky is disabled for given window size
      sidebar_adjust = 0; // vertical offset within sidebar when sticky

  // function to set style for given state
  function set_style(target, value, adjust)
  {
    if(value <= adjust || sticky_disabled){
      target.css({marginLeft: "", position: "", top: "", left: "", bottom: ""});
    }
    else if (value <= holder.height() - target.height() + adjust){
      target.css({marginLeft: 0, position: "fixed", top: - adjust - topFrameOffset(),
                 left: leftViewOffset(holder), bottom: ""});
    }
    else{
      target.css({marginLeft: 0, position: "absolute", top: "", left: 0, bottom: 0});
    }
  }

  // func to update sidebar position based on scrollbar & container positions
  function update_sticky(){
    // set sidebar position
    var offset = -topViewOffset(holder, true);
    set_style(sidebar, offset, sidebar_adjust);
    // collapse button should follow along as well
    set_style(toggle, offset, 0);
  };

  // func to update sidebar measurements, and then call update_sticky()
  function update_measurements(){
    sticky_disabled = false;
    sidebar_adjust = 0;
    if(toc_header){
      // check how much room we have to display top of sidebar -> end of toc list
      var leftover = $(window).height() - (toc_list.height() + topViewOffset(toc_list) - topViewOffset(sidebar));
      if(leftover < 0){
        // not enough room if we align top of sidebar to window,
        // try aligning to top of toc list instead
        sidebar_adjust = topViewOffset(toc_header) - topViewOffset(sidebar) - 8;
        if(leftover + sidebar_adjust < 0){
          // still not enough room - disable sticky sidebar
          sticky_disabled = true;
        }
      }
    }
    update_sticky();
  }

  // run function now, and every time window scrolls
  update_measurements();
  $(window).scroll(update_sticky)
           .resize(update_measurements)
           .bind('hashchange', update_measurements)
           .bind('cloud-section-toggled', update_measurements);
});


/* ==========================================================================
 * sidebar toc highlighter
 * ==========================================================================
 *
 * highlights toc entry for current section being viewed.
 */
$(document).ready(function (){
  // scan & init all links w/in local & global toc
  var $links = $(".sphinxlocaltoc ul a, .sphinxglobaltoc li.current a");
  $links = $links.filter(function () {
    // grab basic info about link
    var $link = $(this),
        href = shorten_url($link.prop("href")),
        $item = $link.parent("li"),
        $parent = $item.parent("ul").prev("a"),
        parent_entry = $parent.data("toc"),
        $target;

    // handle the various types of links
    if (!href || href[0] == "#") {
      // link points to section in current document.
      // use that section as visibility target
      $target = href ? $(href).first() : $("h1").parent();
      $link.addClass("local");

    } else if (!parent_entry) {
      // parent link isn't local, so we can ignore this link
      return false;

    } else {
      // parent is part of page, so this is link for child page.
      // prefer to use actual link in document as visibility target
      $target = parent_entry.target.find("a").filter(function (){ return shorten_url(this.href) == href; });
      if (!$target.length) {
        // fall back to parent section
        $target = parent_entry.target;
      } else if ($target.parent("li").attr("class").search(/(^|\w)toctree-/) == 0) {
        // it's part of embedded toc tree, use whole item.
        $target = $target.parent("li");
      }
      $link.addClass("child");
    }

    // add entry for link containing pre-cached ref to all the data highlighter needs.
    var $ul = $link.next("ul");
    if (!href || !$ul.length || $ul.find("a").length < 4) { $ul = null; }
    else { $item.addClass("toc-toggle"); }
    $link.data("toc", {target: $target, // section/object controlling link's visibility
                       item: $item, // list item where we set state flags
                       child_ul: $ul, // list of child items if we're doing collapse checking, else null
                       parent: $parent.length ? $parent : null, // parent link, if any, else null
                       parent_entry: parent_entry, // direct reference to parent entry
                       first_child_target: null, // first child $target, if any
                       active_child: null // first active child link, if any -- updated each rebuild
                       });

    // set first_child on parent entry, if needed.
    if (parent_entry && !parent_entry.first_child_target && !$link.hasClass("child")) {
      parent_entry.first_child_target = $target;
    }
    return true;
  });

  // if all links got filtered, nothing to do.
  if(!$links.length) return;

  // debugging helper
//  var $cutoff = $('<div/>', {id: "cutoff", style:'position: fixed; left: 0; right: 0; background: rgba(255,0,0,0.1); '}).appendTo("body");

  // function to update toc markers
  function update_visible_sections(evt, first_run){
    // determine viewable range -- make this call once for speed
    var height = $(window).height(),
        line_height = csspx(".body", "line-height"),
        topCutoff = Math.floor(Math.min(height * 0.15, 5 * line_height));
//        bottomCutoff = height - Math.floor(Math.min(height * 0.5, 30 * line_height));
//    $cutoff.css({height: topCutoff, top: 0});
//    $cutoff.css({height: height - bottomCutoff, bottom: 0});

    // update flags for all link elements.
    for (var i=0; i < $links.length; ++i) {
      // grab link element & work out state
      var $link = $($links[i]),
          entry = $link.data("toc"),
          parent_entry = entry.parent_entry,
          $target = entry.target,
          $li = entry.item,
            // viewport-relative position of target
          rect = $target[0].getBoundingClientRect(), // NOTE: width/height not available in IE8-
            // whether section contents are visible
          target_hidden = (rect.bottom < 0 || rect.top > height),
          hidden = target_hidden || $target.hasClass("collapsed"),
            // whether this is the 'active' section
          active = (!hidden && (!parent_entry ||
                                (parent_entry.item.hasClass("active") &&
                                !parent_entry.active_child &&
                                topViewOffset(parent_entry.first_child_target) < topCutoff)) &&
                    rect.bottom >= topCutoff && !$link.hasClass("child"));

      // update active status
      entry.active_child = null;
      if (parent_entry && active) {
        parent_entry.active_child = $link;
        parent_entry.item.removeClass("final");
      }
      $li.toggleClass("active", active);
      $li.toggleClass("final", active);

// NOTE: works, but 'visible' flag not used
/*      // update visible status -- unlike active, excludes child content.
      // NOTE: using 'target_hidden' so collapsed sections still show up
      $li.toggleClass("visible", !target_hidden &&
                      topViewOffset(entry.first_child_target) >= 0); */

      // update collapsed status of child entries
      var $ul = entry.child_ul;
      if ($ul){
        // XXX: this logic open sections when they're visible,
        //      but creates a lot of motion in the TOC.
//        var collapsed = hidden || (!active && (rect.bottom < topCutoff || rect.top > bottomCutoff));
        var collapsed = hidden || !active;
        entry.just_opened = false;
        if($li.hasClass("collapsed") != collapsed) {
          if(!collapsed) {
            if(parent_entry && parent_entry.just_opened) {
              $ul.show(); // don't animate if parent is animated.
            }else {
              $ul.slideDown();
            }
            entry.just_opened = true;
          } else if (first_run) {
            $ul.hide(); // don't animate things on first run
          } else {
            $ul.slideUp();
          }
          $li.toggleClass("collapsed", collapsed);
        }
      }
    }
  }

  // run function now, and every time window is resized
  // TODO: disable when sidebar isn't sticky (including when window is too small)
  //       and when sidebar is collapsed / invisible
  update_visible_sections(null, true);
  $(window).scroll(update_visible_sections)
           .resize(update_visible_sections)
           .bind('hashchange', update_visible_sections)
           .bind('cloud-section-toggled', update_visible_sections)
           .bind('cloud-sidebar-toggled', update_visible_sections);
});


/* ==========================================================================
 * header breaker
 * ==========================================================================
 *
 * attempts to intelligently insert linebreaks into page titles, where possible.
 * currently only handles titles such as "module - description",
 * adding a break after the "-".
 */
$(document).ready(function (){
  // get header's content, insert linebreaks
  var header = $("h1");
  var orig = header[0].innerHTML;
  var shorter = orig;
  if($("h1 > a:first > tt > span.pre").length > 0){
      shorter = orig.replace(/(<\/tt><\/a>\s*[-\u2013\u2014:]\s+)/im, "$1<br> ");
  }
  else if($("h1 > tt.literal:first").length > 0){
      shorter = orig.replace(/(<\/tt>\s*[-\u2013\u2014:]\s+)/im, "$1<br> ");
  }
  if(shorter == orig){
    return;
  }

  // hack to determine full width of header
  header.css({whiteSpace: "nowrap", position:"absolute"});
  var header_width = header.width();
  header.css({whiteSpace: "", position: ""});

  // func to insert linebreaks when needed
  function layout_header(){
    header[0].innerHTML = (header_width > header.parent().width()) ? shorter : orig;
  }

  // run function now, and every time window is resized
  layout_header();
  $(window).resize(layout_header)
           .bind('cloud-sidebar-toggled', layout_header);
});


/* ==========================================================================
 * toc cleaner
 * ==========================================================================
 *
 * attempts to remove clutter from toc lists.
 * mainly, looks for toc entries with format "{module} -- {desc}"",
 * and reduces them down to just "" to save space.
 */
$(document).ready(function (){
  var $toc = $(".sphinxglobaltoc"),
      candidates = {};

  // scan TOC for module entries
  $toc.find("a.internal.reference:has(.literal:first-child > .pre)").each(function (){
    var $this = $(this),
        text = $this.find(".literal").text();

        // wrap details in .objdesc class
        $this.contents().filter(function (){
          return ((this.nodeType == TEXT_NODE))
        }).wrap('<span class="objdesc"></span>');

        // work out modules that have exactly 1 toc entry
        if (isUndef(candidates[text])) {
          candidates[text] = $this;
        }else{
          candidates[text] = null;
        }
  });

  // for all modules that had unique names, add 'unique' flag
  $.each(candidates, function (text, $entry){
    if ($entry) { $entry.children(".objdesc").addClass("unique"); }
  });

});



/* ==========================================================================
 * auto determine when admonition should have inline / block title
 * under this mode, the css will default to styling everything like a block,
 * so we just mark everything that shouldn't be blocked out.
 * ==========================================================================
 */
$(document).ready(function (){
  $("div.body div.admonition:not(.inline-title):not(.block-title)" +
                           ":not(.danger):not(.error)" +
                           ":has(p.first + p.last)").addClass("inline-title");
});


/* ==========================================================================
 * codeblock lineno aligner
 * if document contains multiple codeblocks, and some have different counts
 * (e.g. 10 lines vs 300 lines), the alignment will look off, since the
 * 300 line block will be indented 1 extra space to account for the hundreds.
 * this unifies the widths of all such blocks (issue 19)
 * ==========================================================================
 */
$(document).ready(function (){
  var $lines = $(".linenodiv pre");
  if(!$lines.length) { return; }
  // NOTE: using ems so this holds under font size changes
  var largest = Math.max.apply(null, $lines.map(function () { return $(this).innerWidth(); })),
      em_to_px = csspx($lines, "font-size");
  $lines.css("width", (largest / em_to_px) + "em").css("text-align", "right");
});

/* ==========================================================================
 * codeblock copy helper button
 * ==========================================================================
 *
 * Add a [>>>] button on the top-right corner of code samples to hide
 * the '>>>' and '...' prompts and the output and thus make the code
 * copyable. Also hides linenumbers.
 *
 * Adapted from copybutton.js,
 * Copyright 2014 PSF. Licensed under the PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
 * File originates from the cpython source found in Doc/tools/sphinxext/static/copybutton.js
 *
 */
$(document).ready(function() {
  // TODO: enhance this to hide linenos for ALL highlighted code blocks,
  //       and only perform python-specific hiding as-needed.

  // static text
  var hide_text = 'Hide the prompts and output',
      show_text = 'Show the prompts and output';

  // helper which sets button & codeblock state
  function setButtonState($button, active) {
    $button.parent().find('.go, .gp, .gt').toggle(!active);
    $button.next('pre').find('.gt').nextUntil('.gp, .go').css('visibility', active ? 'hidden' : 'visible');
    $button.closest(".highlighttable").find(".linenos pre").css('visibility', active ? 'hidden' : 'visible');
    $button.attr('title', active ? show_text : hide_text);
    $button.toggleClass("active", active);
  }

  // create and add the button to all the code blocks containing a python prompt
  var $blocks = $('.highlight-python, .highlight-python3');
  $blocks.find(".highlight:has(pre .gp)").each(function() {
    var $block = $(this);

    // tracebacks (.gt) contain bare text elements that need to be
    // wrapped in a span to work with .nextUntil() call in setButtonState()
    $block.find('pre:has(.gt)').contents().filter(function() {
      return ((this.nodeType == TEXT_NODE) && (this.data.trim().length > 0));
    }).wrap('<span>');

    // insert button into block
    var $button = $('<button class="copybutton">&gt;&gt;&gt;</button>');
    $block.css("position", "relative").prepend($button);
    setButtonState($button, false);
  });

  // toggle button state when clicked
  $('.copybutton').click(
    function() {
      var $button = $(this);
      setButtonState($button, !$button.hasClass("active"));
    });
});

/* ==========================================================================
 * eof
 * ==========================================================================
 */

// end encapsulation
// NOTE: sphinx provides underscore.js as $u
}(window, jQuery, $u));
